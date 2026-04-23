import time
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    rs = None


class RealSenseManager:
    """
    Robust RealSense manager with:
      - supported profile negotiation (USB2/USB3 friendly)
      - warm-up, longer timeouts, queue sizing
      - safe alignment (checks frames; falls back if missing)
      - per-device failure counters with auto-restart
    """

    def __init__(
        self,
        desired_width=640,
        desired_height=480,
        desired_fps=15,
        warmup_frames=30,
        wait_timeout_ms=1000,
        device_serials=None,   # list[str] or None => all devices
        align_to_color=True,
        prefer_mjpeg=False,     # try MJPEG color first (reduces bandwidth on USB2)
        verbose=True,
    ):
        self.enabled = REALSENSE_AVAILABLE
        self.verbose = verbose
        if not self.enabled:
            print("⚠️ pyrealsense2 not found; RealSense disabled.")
            return

        self.desired_w = int(desired_width)
        self.desired_h = int(desired_height)
        self.desired_fps = int(desired_fps)
        self.warmup_frames = int(warmup_frames)
        self.wait_timeout_ms = int(wait_timeout_ms)
        self.device_serials = device_serials  # None = all
        self.align_to_color = align_to_color
        self.prefer_mjpeg = prefer_mjpeg

        self.ctx = rs.context()
        self.devices_info = {}       # name -> {serial, name, index}
        self.pipelines = {}          # name -> rs.pipeline
        self.aligners = {}           # name -> rs.align
        self.pipeline_profiles = {}  # name -> rs.pipeline_profile

        # failure handling
        self._fail_counts = {}           # name -> consecutive failures
        self._fail_restart_thresh = 5    # restart after N consecutive failures

        self._discover_devices()
        if self.devices_info:
            self._start_all()
        else:
            self.enabled = False

    # ---------- Public API ----------

    def is_enabled(self):
        return self.enabled

    def stop(self):
        for name, pipe in self.pipelines.items():
            try:
                pipe.stop()
                if self.verbose:
                    print(f"🛑 Stopped {name}")
            except Exception:
                pass

    def capture_frames(self, img_size=None):
        if not self.enabled:
            return {}

        out = {}
        to_restart = []

        for name, pipe in self.pipelines.items():
            pack = self._capture_one(name, pipe, img_size)
            if pack is None:
                # failure handling
                self._fail_counts[name] = self._fail_counts.get(name, 0) + 1
                if self._fail_counts[name] >= self._fail_restart_thresh:
                    if self.verbose:
                        print(f"♻️  Restarting pipeline {name} (consecutive failures={self._fail_counts[name]})")
                    to_restart.append(name)
            else:
                # success → reset failure counter
                self._fail_counts[name] = 0
                out[name] = pack

        # Restart any flaky pipelines
        for name in to_restart:
            try:
                self._restart_one(name)
            except Exception as e:
                if self.verbose:
                    print(f"❌ Restart {name} failed: {e}")

        return out

    # ---------- Internal: one-device capture ----------

    def _capture_one(self, name, pipe, img_size=None):
        frameset = self._wait_for_frames(pipe, self.wait_timeout_ms)
        if frameset is None:
            frameset = pipe.poll_for_frames()
            if frameset is None:
                if self.verbose:
                    print(f"⚠️ {name}: no frames this cycle")
                return None

        # Quick check for presence before alignment
        depth0 = frameset.get_depth_frame()
        color0 = frameset.get_color_frame()
        if not depth0 or not color0:
            # try a second wait (transient)
            frameset2 = self._wait_for_frames(pipe, self.wait_timeout_ms)
            if frameset2:
                frameset = frameset2
                depth0 = frameset.get_depth_frame()
                color0 = frameset.get_color_frame()

        # Align only if both are present and we have an aligner
        aligned = False
        if self.align_to_color and name in self.aligners and depth0 and color0:
            try:
                frameset_aligned = self.aligners[name].process(frameset)
                # verify aligned frames exist
                depth = frameset_aligned.get_depth_frame()
                color = frameset_aligned.get_color_frame()
                if depth and color:
                    frameset = frameset_aligned
                    aligned = True
                else:
                    if self.verbose:
                        print(f"⚠️ {name}: aligned frames missing, using unaligned.")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ {name}: align failed ({e}); using unaligned.")
                # fall through with unaligned frameset

        # Final grab (unaligned or aligned)
        depth = frameset.get_depth_frame()
        color = frameset.get_color_frame()
        if not depth or not color:
            if self.verbose:
                print(f"⚠️ {name}: missing depth/color after{' aligned' if aligned else ''} capture")
            return None

        # Numpy conversions
        depth_np = np.asanyarray(depth.get_data())
        color_np = np.asanyarray(color.get_data())
        if color.get_profile().format() == rs.format.rgb8:
            color_np = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)

        # Display-friendly depth
        depth_display = np.clip(depth_np.astype(np.float32), 0, 5000)
        depth_display = (depth_display / 5000.0 * 255.0).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

        if img_size is not None:
            color_np = cv2.resize(color_np, img_size)
            depth_np = cv2.resize(depth_np, img_size)
            depth_colormap = cv2.resize(depth_colormap, img_size)

        return {
            "color": color_np,
            "depth": depth_np,
            "depth_colormap": depth_colormap,
            "timestamp": time.time(),
        }

    # ---------- Internal: discovery & startup ----------

    def _discover_devices(self):
        try:
            devs = self.ctx.query_devices()
        except Exception as e:
            print(f"❌ query_devices failed: {e}")
            return

        if len(devs) == 0:
            print("⚠️ No RealSense devices found.")
            return

        if self.verbose:
            print(f"🔍 Found {len(devs)} RealSense device(s)")

        for i, dev in enumerate(devs):
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)

            if self.device_serials and serial not in self.device_serials:
                continue
            key = self._device_key(name)
            self.devices_info[key] = {"serial": serial, "name": name, "index": i}
            if self.verbose:
                print(f"  • {key}: {name} (S/N: {serial})")

        if not self.devices_info:
            print("⚠️ No target RealSense devices after filtering.")

    def _start_all(self):
        for name, info in self.devices_info.items():
            ok = self._start_one(name, info["serial"])
            if not ok:
                print(f"❌ Failed to start {name}")
        if self.pipelines:
            self.enabled = True
            if self.verbose:
                print(f"✅ Initialized {len(self.pipelines)} RealSense device(s)")
        else:
            self.enabled = False

    def _start_one(self, dev_name, serial):
        """
        Start a single device with robust profile negotiation and warm-up.
        """
        pipe = rs.pipeline()

        # Try profile candidates in order (prefer MJPEG color if available)
        color_formats = [rs.format.mjpeg, rs.format.bgr8] if self.prefer_mjpeg else [rs.format.bgr8, rs.format.mjpeg]
        whfps_candidates = [
            (self.desired_w, self.desired_h, self.desired_fps),
            (640, 480, 15),
            (640, 480, 6),
            (848, 480, 15),
            (424, 240, 15),
            (424, 240, 6),
        ]

        last_err = None

        for cf in color_formats:
            for (w, h, fps) in whfps_candidates:
                # try:
                cfg = rs.config()
                cfg.enable_device(serial)
                cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
                cfg.enable_stream(rs.stream.color, w, h, cf, fps)

                # Resolve to ensure combo valid
                cfg.resolve(pipe)
                profile = pipe.start(cfg)

                # bump frames queue
                dev = profile.get_device()
                for s in dev.sensors:
                    try:
                        s.set_option(rs.option.frames_queue_size, 8)
                    except Exception:
                        pass

                # aligner per device
                self.aligners[dev_name] = rs.align(rs.stream.color if self.align_to_color else rs.stream.depth)

                # warm-up frames
                warm_ok = False
                for _ in range(self.warmup_frames):
                    fs = self._wait_for_frames(pipe, self.wait_timeout_ms)
                    if fs:
                        d0 = fs.get_depth_frame()
                        c0 = fs.get_color_frame()
                        if d0 and c0:
                            warm_ok = True
                            break
                if not warm_ok and self.verbose:
                    print(f"⚠️ {dev_name}: warm-up did not get both streams; continuing anyway")

                self.pipelines[dev_name] = pipe
                self.pipeline_profiles[dev_name] = profile
                self._fail_counts[dev_name] = 0

                if self.verbose:
                    cfmt = "MJPEG" if cf == rs.format.mjpeg else ("BGR8" if cf == rs.format.bgr8 else str(cf))
                    print(f"✅ {dev_name} started @ {w}x{h} {fps}fps (color {cfmt}, depth Z16)")
                return True

        print(f"❌ Could not start {dev_name}: {last_err}")
        return False

    def _restart_one(self, name):
        """Stop and restart a single device pipeline with same serial."""
        if name not in self.devices_info:
            return
        serial = self.devices_info[name]["serial"]
        # stop old pipeline
        try:
            if name in self.pipelines:
                self.pipelines[name].stop()
        except Exception:
            pass
        time.sleep(0.2)
        # remove old entries
        self.pipelines.pop(name, None)
        self.pipeline_profiles.pop(name, None)
        self.aligners.pop(name, None)
        # start again
        ok = self._start_one(name, serial)
        if ok:
            self._fail_counts[name] = 0

    # ---------- Internal helpers ----------

    def _wait_for_frames(self, pipeline, timeout_ms):
        try:
            return pipeline.wait_for_frames(timeout_ms=timeout_ms)
        except Exception:
            return None

    @staticmethod
    def _device_key(name: str) -> str:
        n = (name or "").lower()
        if "d435i" in n or "d430i" in n:
            return "d435i"
        if "d435" in n:
            return "d435"
        if "realsense" in n:
            return "realsense"
        return n or "device"

