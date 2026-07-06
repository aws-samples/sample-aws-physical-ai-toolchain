#!/usr/bin/env python3.10
"""
Keyboard teleoperation for UR3 simulator (SSH tunnel compatible).

Opens a browser page that reads keyboard input via WASD+IJKL (and
optionally a gamepad), then sends movel commands to the robot using
ephemeral sockets that work through SSH tunnels.

Usage:
    python3 examples/keyboard_teleop.py
    python3 examples/keyboard_teleop.py --record "pick up the cube"
    python3 examples/keyboard_teleop.py --port 8765

SSH tunnel setup (forward these ports from your Mac to EC2):
    ssh -L 30002:localhost:30002 -L 30003:localhost:30003 \
        -L 29999:localhost:29999 -L 8554:localhost:8554 ec2-host

Controls (keyboard):
    W/A/S/D           Move in X/Y plane
    I/K               Move up/down (Z)
    J/L               Rotate wrist
    Q                 Open gripper
    E                 Close gripper
    H                 Reset to home position
    Shift             Precision mode (hold)
    Space             Fast mode (hold)
    R                 Start/stop recording
    X                 Emergency stop
    Escape            Quit

Controls (gamepad -- if connected):
    Left Stick        Move in X/Y plane
    Right Stick Y     Move up/down (Z)
    Right Stick X     Rotate wrist
    Left Trigger      Open gripper
    Right Trigger     Close gripper
    Y                 Reset to home position
    Left Bumper       Precision mode
    Right Bumper      Fast mode
    A                 Start/stop recording
    B                 Emergency stop
    Start / Menu      Quit

No extra dependencies -- uses Python stdlib only (plus safe_controller).
"""

import http.server
import json
import math
import os
import socket
import sys
import threading
import time
import webbrowser

# -- Config -------------------------------------------------------------------

ROBOT_IP = os.environ.get("ROBOT_IP", "127.0.0.1")
CAMERA_URL = os.environ.get("CAMERA_URL", "http://localhost:8554/")
START_JOINTS_DEG = [38.5, -90.4, 78.5, -90.2, -92.7, 12.5]

SPEED_NORMAL = 0.08
SPEED_SLOW = 0.02
SPEED_FAST = 0.15
ROT_SPEED = 0.3

DEADZONE = 0.15
LOOP_HZ = 10
MAX_REACH = 0.42
MIN_Z = 0.03
MAX_Z = 0.55

# -- HTML page (embedded) ----------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>UR3 Teleop (Sim)</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, 'Segoe UI', monospace; background: #1a1a2e; color: #e0e0e0; padding: 20px; }
  h1 { color: #00d4ff; margin-bottom: 10px; font-size: 1.4em; }
  .status { padding: 8px 14px; border-radius: 6px; margin-bottom: 16px; font-weight: 600; }
  .status.ok { background: #0a3d0a; color: #4caf50; border: 1px solid #4caf50; }
  .status.warn { background: #3d3d0a; color: #ffeb3b; border: 1px solid #ffeb3b; }
  .status.err { background: #3d0a0a; color: #f44336; border: 1px solid #f44336; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .card { background: #16213e; border: 1px solid #334; border-radius: 8px; padding: 14px; }
  .card h2 { color: #00d4ff; font-size: 0.9em; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }
  .val { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 1.1em; line-height: 1.8; }
  .val span { color: #888; }
  .bar-row { display: flex; align-items: center; gap: 8px; margin: 3px 0; }
  .bar-label { width: 24px; text-align: right; color: #888; font-size: 0.85em; }
  .bar-track { flex: 1; height: 14px; background: #0f3460; border-radius: 7px; position: relative; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 7px; transition: width 0.05s; }
  .bar-fill.pos { background: #00d4ff; }
  .bar-fill.neg { background: #e94560; }
  .bar-center { position: absolute; left: 50%; top: 0; bottom: 0; width: 2px; background: #555; }
  .btn-row { display: flex; gap: 6px; flex-wrap: wrap; }
  .btn { padding: 4px 10px; border-radius: 4px; font-size: 0.8em; background: #0f3460; border: 1px solid #334; }
  .btn.on { background: #00d4ff; color: #000; font-weight: 700; border-color: #00d4ff; }
  .controls { font-size: 0.85em; line-height: 1.7; color: #999; }
  .controls b { color: #ddd; }
  .rec { color: #f44336; font-weight: 700; animation: blink 1s infinite; }
  @keyframes blink { 50% { opacity: 0.3; } }
  .tcp { font-family: 'SF Mono', monospace; font-size: 1.05em; }
</style>
</head>
<body>
<h1>UR3 Teleop (Simulator)</h1>
<div id="status" class="status warn">Press any key to start...</div>

<div class="grid">
  <div class="card">
    <h2>Input</h2>
    <div id="sticks">
      <div class="bar-row"><span class="bar-label">LX</span><div class="bar-track"><div class="bar-center"></div><div id="bar-lx" class="bar-fill pos" style="width:50%"></div></div></div>
      <div class="bar-row"><span class="bar-label">LY</span><div class="bar-track"><div class="bar-center"></div><div id="bar-ly" class="bar-fill pos" style="width:50%"></div></div></div>
      <div class="bar-row"><span class="bar-label">RX</span><div class="bar-track"><div class="bar-center"></div><div id="bar-rx" class="bar-fill pos" style="width:50%"></div></div></div>
      <div class="bar-row"><span class="bar-label">RY</span><div class="bar-track"><div class="bar-center"></div><div id="bar-ry" class="bar-fill pos" style="width:50%"></div></div></div>
      <div class="bar-row"><span class="bar-label">LT</span><div class="bar-track"><div id="bar-lt" class="bar-fill pos" style="width:0%"></div></div></div>
      <div class="bar-row"><span class="bar-label">RT</span><div class="bar-track"><div id="bar-rt" class="bar-fill pos" style="width:0%"></div></div></div>
    </div>
    <div id="buttons" class="btn-row" style="margin-top:10px"></div>
  </div>
  <div class="card">
    <h2>Robot</h2>
    <div id="robot-info" class="val">Waiting for data...</div>
  </div>
</div>

<div class="card controls">
  <b>Keyboard:</b> <b>WASD</b> X/Y &nbsp; <b>IK</b> Z up/down &nbsp; <b>JL</b> Rotate &nbsp; <b>Q</b> Open &nbsp; <b>E</b> Close &nbsp; <b>H</b> Home &nbsp; <b>Shift</b> Slow &nbsp; <b>Space</b> Fast &nbsp; <b>R</b> Rec &nbsp; <b>X</b> E-stop &nbsp; <b>Esc</b> Quit<br>
  <b>Gamepad:</b> Left Stick X/Y &nbsp; Right Stick Z/Rot &nbsp; LT Open &nbsp; RT Close &nbsp; Y Home &nbsp; LB Slow &nbsp; RB Fast &nbsp; A Rec &nbsp; B E-stop
</div>

<script>
let connected = false;
let kbActive = false;
let lastSend = 0;
const SEND_INTERVAL = 100;

const keys = {};
document.addEventListener('keydown', (e) => {
  keys[e.key.toLowerCase()] = true;
  if (e.key === ' ') e.preventDefault();
  if (!kbActive && !connected) {
    kbActive = true;
    document.getElementById('status').className = 'status ok';
    document.getElementById('status').textContent = 'Keyboard active (WASD + IJKL)';
    requestAnimationFrame(pollInput);
  }
});
document.addEventListener('keyup', (e) => {
  keys[e.key.toLowerCase()] = false;
});
document.addEventListener('keydown', (e) => { if (e.shiftKey) keys['shift'] = true; });
document.addEventListener('keyup', (e) => { if (!e.shiftKey) keys['shift'] = false; });

function kbAxis(neg, pos) {
  return (keys[pos] ? 1 : 0) - (keys[neg] ? 1 : 0);
}

function updateBar(id, value, centered) {
  const el = document.getElementById(id);
  if (centered) {
    if (value >= 0) {
      el.style.marginLeft = '50%';
      el.style.width = (value * 50) + '%';
      el.className = 'bar-fill pos';
    } else {
      const w = -value * 50;
      el.style.marginLeft = (50 - w) + '%';
      el.style.width = w + '%';
      el.className = 'bar-fill neg';
    }
  } else {
    el.style.marginLeft = '0';
    el.style.width = (value * 100) + '%';
    el.className = 'bar-fill pos';
  }
}

function pollInput() {
  const gamepads = navigator.getGamepads();
  let gp = null;
  for (const g of gamepads) {
    if (g && g.connected) { gp = g; break; }
  }

  if (gp && !connected) {
    connected = true;
    document.getElementById('status').className = 'status ok';
    document.getElementById('status').textContent = gp.id + ' + Keyboard';
  }
  if (!gp && connected) {
    connected = false;
    if (kbActive) {
      document.getElementById('status').className = 'status ok';
      document.getElementById('status').textContent = 'Keyboard active (WASD + IJKL)';
    } else {
      document.getElementById('status').className = 'status warn';
      document.getElementById('status').textContent = 'Press any key to start...';
    }
  }

  const gLx = gp ? (gp.axes[0] || 0) : 0;
  const gLy = gp ? (gp.axes[1] || 0) : 0;
  const gRx = gp ? (gp.axes[2] || 0) : 0;
  const gRy = gp ? (gp.axes[3] || 0) : 0;
  const gLt = gp && gp.buttons[6] ? gp.buttons[6].value : 0;
  const gRt = gp && gp.buttons[7] ? gp.buttons[7].value : 0;
  const gA  = gp && gp.buttons[0] ? gp.buttons[0].pressed : false;
  const gB  = gp && gp.buttons[1] ? gp.buttons[1].pressed : false;
  const gLb = gp && gp.buttons[4] ? gp.buttons[4].pressed : false;
  const gRb = gp && gp.buttons[5] ? gp.buttons[5].pressed : false;
  const gY  = gp && gp.buttons[3] ? gp.buttons[3].pressed : false;
  const gSt = gp && gp.buttons[9] ? gp.buttons[9].pressed : false;

  const kLx = kbAxis('a', 'd');
  const kLy = kbAxis('w', 's');
  const kRx = kbAxis('j', 'l');
  const kRy = kbAxis('i', 'k');
  const kLt = keys['q'] ? 1 : 0;
  const kRt = keys['e'] ? 1 : 0;
  const kA  = !!keys['r'];
  const kB  = !!keys['x'];
  const kLb = !!keys['shift'];
  const kRb = !!keys[' '];
  const kY  = !!keys['h'];
  const kSt = !!keys['escape'];

  const lx = Math.abs(gLx) > 0.1 ? gLx : kLx;
  const ly = Math.abs(gLy) > 0.1 ? gLy : kLy;
  const rx = Math.abs(gRx) > 0.1 ? gRx : kRx;
  const ry = Math.abs(gRy) > 0.1 ? gRy : kRy;
  const lt = Math.max(gLt, kLt);
  const rt = Math.max(gRt, kRt);

  updateBar('bar-lx', lx, true);
  updateBar('bar-ly', ly, true);
  updateBar('bar-rx', rx, true);
  updateBar('bar-ry', ry, true);
  updateBar('bar-lt', lt, false);
  updateBar('bar-rt', rt, false);

  const btnState = [
    ['W', keys['w']], ['A', keys['a']], ['S', keys['s']], ['D', keys['d']],
    ['I', keys['i']], ['J', keys['j']], ['K', keys['k']], ['L', keys['l']],
    ['Q', keys['q']], ['E', keys['e']], ['Shift', keys['shift']], ['Space', keys[' ']],
    ['R', keys['r']], ['X', keys['x']],
  ];
  if (gp) {
    const gpNames = ['A','B','X','Y','LB','RB','LT','RT','Back','Start'];
    for (let i = 0; i < gp.buttons.length && i < gpNames.length; i++) {
      btnState.push([gpNames[i], gp.buttons[i].pressed]);
    }
  }
  let html = '';
  for (const [name, on] of btnState) {
    html += '<span class="btn' + (on ? ' on' : '') + '">' + name + '</span>';
  }
  document.getElementById('buttons').innerHTML = html;

  const now = performance.now();
  if ((connected || kbActive) && now - lastSend > SEND_INTERVAL) {
    lastSend = now;
    const state = {
      lx: lx, ly: ly, rx: rx, ry: ry,
      lt: lt, rt: rt,
      btn_a: gA || kA,
      btn_b: gB || kB,
      btn_y: gY || kY,
      btn_lb: gLb || kLb,
      btn_rb: gRb || kRb,
      btn_start: gSt || kSt,
    };
    fetch('/state', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(state),
    }).then(r => r.json()).then(data => {
      let info = '<div class="tcp">';
      if (data.tcp) {
        const t = data.tcp;
        info += '<span style="color:#888">pos</span> x=' + t[0].toFixed(3) + ' y=' + t[1].toFixed(3) + ' z=' + t[2].toFixed(3) + '<br>';
        info += '<span style="color:#888">rot</span> rx=' + t[3].toFixed(3) + ' ry=' + t[4].toFixed(3) + ' rz=' + t[5].toFixed(3) + '<br>';
      }
      if (data.joints) {
        const j = data.joints;
        const deg = j.map(r => (r * 180 / Math.PI).toFixed(1));
        info += '<span style="color:#888">joints</span> ' + deg.join(' / ') + '&deg;<br>';
      }
      info += '<span>' + (data.speed_label || '') + '</span>';
      info += ' &nbsp; grip=' + (data.gripper || '?');
      if (data.recording) info += ' &nbsp; <span class="rec">REC</span>';
      if (data.message) info += '<br><span style="color:#ffeb3b">' + data.message + '</span>';
      info += '</div>';
      document.getElementById('robot-info').innerHTML = info;
    }).catch(() => {});
  }

  requestAnimationFrame(pollInput);
}

window.addEventListener("gamepadconnected", () => { if (!kbActive) pollInput(); });
requestAnimationFrame(pollInput);
</script>
</body>
</html>
"""

# -- Robot helpers ------------------------------------------------------------

def apply_deadzone(value):
    if abs(value) < DEADZONE:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - DEADZONE) / (1.0 - DEADZONE)


class URScriptSender:
    """Send URScript commands via a persistent socket to port 30002.

    Port 30002 sends robot state data back.  A drain thread continuously
    reads and discards that data so the TCP receive buffer never fills
    and back-pressures our sends.

    speedl with t=0 is non-blocking in URScript — it sets the target
    velocity and returns instantly, so commands never queue.
    """

    def __init__(self, host, port=30002):
        self.host = host
        self.port = port
        self._sock = None
        self._running = False

    def connect(self):
        self.close()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(2.0)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.connect((self.host, self.port))
        self._running = True
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        """Read and discard incoming state data to prevent buffer backpressure."""
        while self._running and self._sock:
            try:
                data = self._sock.recv(4096)
                if not data:
                    break
            except (socket.timeout, OSError):
                continue

    def send(self, program):
        try:
            self._sock.sendall(program.encode("utf-8"))
        except (socket.error, OSError, AttributeError):
            self.connect()
            self._sock.sendall(program.encode("utf-8"))

    def send_speedl(self, vel, accel=1.0):
        """Velocity command wrapped in a program block (required by port 30002)."""
        v = f"[{vel[0]:.5f},{vel[1]:.5f},{vel[2]:.5f},{vel[3]:.5f},{vel[4]:.5f},{vel[5]:.5f}]"
        self.send(f"def cmd():\n  speedl({v},{accel},0.12)\nend\n")

    def send_stopj(self):
        self.send("def cmd():\n  stopj(3.0)\nend\n")

    def send_gripper(self, position):
        """Control gripper via Modbus URScript (position: 0=open, 255=closed)."""
        self.send(
            "def prog():\n"
            '  socket_open("127.0.0.1", 63352, "g")\n'
            f'  socket_set_var("POS", {position}, "g")\n'
            '  socket_set_var("GTO", 1, "g")\n'
            "  sleep(0.5)\n"
            '  socket_close("g")\n'
            "end\n"
        )

    def close(self):
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


def clamp_to_workspace(tcp, vx, vy, vz):
    if tcp is None:
        return vx, vy, vz
    x, y, z = tcp[0], tcp[1], tcp[2]
    reach = math.sqrt(x * x + y * y)
    if reach > MAX_REACH and reach > 0:
        if (x * vx + y * vy) / reach > 0:
            vx, vy = 0.0, 0.0
    if z <= MIN_Z and vz < 0:
        vz = 0.0
    if z >= MAX_Z and vz > 0:
        vz = 0.0
    return vx, vy, vz


# -- Server -------------------------------------------------------------------

class TeleopState:
    """Shared state between HTTP server and robot control."""
    def __init__(self):
        self.lock = threading.Lock()
        self.gamepad = None
        self.gamepad_time = 0.0
        self.tcp = None
        self.joints = None
        self.gripper = "open"
        self.speed_label = ""
        self.recording = False
        self.message = ""
        self.quit = False
        self.was_moving = False
        self.gripper_until = 0.0
        self.a_was_pressed = False
        self.y_was_pressed = False
        self.robot = None
        self.recorder = None


shared = TeleopState()


class TeleopHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/state":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self.send_error(400)
                return

            with shared.lock:
                shared.gamepad = data
                shared.gamepad_time = time.monotonic()

            with shared.lock:
                resp = {
                    "tcp": shared.tcp,
                    "joints": shared.joints,
                    "gripper": shared.gripper,
                    "speed_label": shared.speed_label,
                    "recording": shared.recording,
                    "message": shared.message,
                }
                shared.message = ""

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


def robot_loop():
    """Main robot control loop -- reads shared state, sends movel commands."""
    from robot.ur3.safe_controller import SafeUR3Controller

    print("Connecting to robot (allow a few seconds for sim bridge)...")
    robot = SafeUR3Controller(ROBOT_IP)
    robot.connect()
    shared.robot = robot

    print("Moving to start position...")
    start_rad = [math.radians(d) for d in START_JOINTS_DEG]
    robot.move_joints(start_rad, vel=0.3, accel=0.3)
    print("Ready.\n")

    sender = URScriptSender(ROBOT_IP)
    sender.connect()
    loop_period = 1.0 / LOOP_HZ

    while not shared.quit:
        t0 = time.monotonic()

        with shared.lock:
            gp = shared.gamepad
            gp_age = t0 - shared.gamepad_time

        tcp = robot.get_tcp_pose()
        joints = robot.get_joints()
        with shared.lock:
            shared.tcp = tcp
            shared.joints = joints

        if gp is None or gp_age > 0.3:
            if shared.was_moving:
                try:
                    sender.send_stopj()
                except (socket.error, OSError):
                    pass
                shared.was_moving = False
            time.sleep(loop_period)
            continue

        # Sticks
        lx = apply_deadzone(gp.get("lx", 0))
        ly = apply_deadzone(gp.get("ly", 0))
        rx = apply_deadzone(gp.get("rx", 0))
        ry = apply_deadzone(gp.get("ry", 0))

        # Speed
        speed = SPEED_NORMAL
        speed_label = ""
        if gp.get("btn_lb"):
            speed = SPEED_SLOW
            speed_label = "SLOW"
        elif gp.get("btn_rb"):
            speed = SPEED_FAST
            speed_label = "FAST"

        with shared.lock:
            shared.speed_label = speed_label

        # Gripper -- skip movel while gripper command is in flight
        gripper_active = t0 < shared.gripper_until
        if gp.get("rt", 0) > 0.5 and shared.gripper != "closed":
            print("  Closing gripper...")
            try:
                sender.send_gripper(255)
                shared.gripper_until = time.monotonic() + 1.5
                print("  Gripper closed.")
            except Exception as e:
                print(f"  Gripper close error: {e}")
            with shared.lock:
                shared.gripper = "closed"
                if shared.recording and shared.recorder:
                    shared.recorder.record_gripper_event(255)
            gripper_active = True

        elif gp.get("lt", 0) > 0.5 and shared.gripper != "open":
            print("  Opening gripper...")
            try:
                sender.send_gripper(0)
                shared.gripper_until = time.monotonic() + 1.5
                print("  Gripper opened.")
            except Exception as e:
                print(f"  Gripper open error: {e}")
            with shared.lock:
                shared.gripper = "open"
                if shared.recording and shared.recorder:
                    shared.recorder.record_gripper_event(0)
            gripper_active = True

        # Movement via speedl -- skip if gripper is active (both use port 30002)
        if not gripper_active:
            vx = -ly * speed
            vy = -lx * speed
            vz = -ry * speed
            wrz = rx * ROT_SPEED

            vx, vy, vz = clamp_to_workspace(tcp, vx, vy, vz)

            moving = abs(vx) > 0.001 or abs(vy) > 0.001 or abs(vz) > 0.001 or abs(wrz) > 0.001
            if moving:
                try:
                    sender.send_speedl([vx, vy, vz, 0, 0, wrz])
                    if not shared.was_moving:
                        print(f"  Moving: vx={vx:.3f} vy={vy:.3f} vz={vz:.3f}")
                    shared.was_moving = True
                except (socket.error, OSError) as e:
                    print(f"  Send error: {e}")
            elif shared.was_moving:
                try:
                    sender.send_stopj()
                except (socket.error, OSError):
                    pass
                shared.was_moving = False

        # Record toggle (R / A, edge-triggered)
        btn_a = gp.get("btn_a", False)
        if btn_a and not shared.a_was_pressed:
            with shared.lock:
                if shared.recorder:
                    if not shared.recording:
                        episode_id = shared.recorder.start()
                        shared.recording = True
                        shared.message = f"RECORDING: {episode_id}"
                        print(f"  RECORDING: {episode_id}")
                    else:
                        metadata = shared.recorder.stop()
                        shared.recording = False
                        shared.message = (f"SAVED: {metadata['episode_id']} "
                                          f"({metadata['telemetry_samples']} samples, "
                                          f"{metadata['camera_frames']} frames)")
                        print(f"  SAVED: {metadata['episode_id']}")
                else:
                    shared.message = "Recording not enabled (use --record)"
        shared.a_was_pressed = btn_a

        # Home reset (Y / H, edge-triggered)
        btn_y = gp.get("btn_y", False)
        if btn_y and not shared.y_was_pressed:
            try:
                sender.send_stopj()
            except (socket.error, OSError):
                pass
            shared.was_moving = False
            with shared.lock:
                shared.message = "Returning to home..."
            print("  Returning to home position...")
            start_rad = [math.radians(d) for d in START_JOINTS_DEG]
            robot.move_joints(start_rad, vel=0.3, accel=0.3)
            with shared.lock:
                shared.message = "Home"
                shared.gripper = "open"
            print("  Home.")
        shared.y_was_pressed = btn_y

        # E-stop (B / X)
        if gp.get("btn_b"):
            try:
                sender.send_stopj()
            except (socket.error, OSError):
                pass
            shared.was_moving = False
            with shared.lock:
                shared.message = "E-STOP"
            print("  E-STOP")

        # Quit (Start / Escape)
        if gp.get("btn_start"):
            shared.quit = True

        elapsed = time.monotonic() - t0
        if elapsed < loop_period:
            time.sleep(loop_period - elapsed)

    # Cleanup
    print("\nStopping...")
    try:
        sender.send_stopj()
    except (socket.error, OSError):
        pass
    sender.close()
    with shared.lock:
        if shared.recording and shared.recorder:
            metadata = shared.recorder.stop()
            print(f"Saved: {metadata['episode_id']}")
    robot.disconnect()


# -- Main ---------------------------------------------------------------------

def main():
    record_task = None
    port = 8765
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--record" and i + 1 < len(args):
            record_task = args[i + 1]
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif args[i] in ("-h", "--help"):
            print(__doc__)
            return
        else:
            print(f"Unknown argument: {args[i]}")
            sys.exit(1)

    # Start robot control thread
    robot_thread = threading.Thread(target=robot_loop, daemon=True)
    robot_thread.start()

    # Wait for robot to connect before setting up recorder
    while shared.robot is None and not shared.quit:
        time.sleep(0.1)

    if record_task and shared.robot:
        from robot.ur3.recorder import EpisodeRecorder
        shared.recorder = EpisodeRecorder(
            robot=shared.robot,
            task_name=record_task.replace(" ", "_"),
            camera_url=CAMERA_URL,
        )

    # Bind to 0.0.0.0 so it's accessible through SSH tunnel port forwarding
    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), TeleopHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    url = f"http://localhost:{port}"
    print(f"Teleop server: {url}")
    print("Opening browser...")
    webbrowser.open(url)
    print()
    print("Controls shown in browser. Press Ctrl+C to quit.")
    print()

    try:
        robot_thread.join()
    except KeyboardInterrupt:
        print()

    shared.quit = True
    server.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
