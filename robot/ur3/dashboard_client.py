"""
UR Dashboard Server client for CB-Series robots.

Communicates with the Dashboard Server via a TCP socket (tunneled through
AWS SSM to localhost:29999). All commands are plain-text; responses are
read back as a single line.

Reference: Dashboard Server CB-Series API (ur-robots.com)
"""

import socket
import time


DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 29999


class DashboardClient:
    """
    Client for the Universal Robots Dashboard Server (CB-Series).

    The Dashboard Server exposes administrative controls—program load/play/stop,
    power management, safety resets, popup control, user roles, logging, and
    system information—over a plain-text TCP interface.

    Usage::

        with DashboardClient() as db:
            print(db.polyscope_version())
            db.load("my_program.urp")
            db.play()

    Or manage the connection manually::

        db = DashboardClient()
        db.connect()
        db.power_on()
        db.disconnect()
    """

    def __init__(self, host: str = DASHBOARD_HOST, port: int = DASHBOARD_PORT, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: socket.socket | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> str:
        """
        Open a TCP connection to the Dashboard Server.

        Returns the welcome banner sent by the server on connect.
        """
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.port))
        # The server immediately sends a welcome line such as
        # "Connected: Universal Robots Dashboard Server"
        return self._recv_line()

    def disconnect(self):
        """Send 'quit' and close the socket."""
        if self._sock is not None:
            try:
                self._send_cmd("quit")
            except Exception:
                pass
            self._sock.close()
            self._sock = None

    def __enter__(self) -> "DashboardClient":
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_cmd(self, cmd: str) -> str:
        """Send *cmd* (without trailing newline) and return the response line."""
        if self._sock is None:
            raise RuntimeError("Not connected. Call connect() first.")
        self._sock.sendall((cmd + "\n").encode("utf-8"))
        return self._recv_line()

    def _recv_line(self) -> str:
        """Read bytes until a newline and return the stripped string."""
        buf = b""
        while True:
            chunk = self._sock.recv(4096)
            if not chunk:
                break
            buf += chunk
            if b"\n" in buf:
                break
        return buf.decode("utf-8", errors="replace").strip()

    # ------------------------------------------------------------------
    # Program control  (available since v1.4)
    # ------------------------------------------------------------------

    def load(self, program_path: str) -> str:
        """
        Load a .urp program (and its associated installation).

        Blocks until loading completes or fails. Fails if the associated
        installation requires safety confirmation.

        Returns the server response string, e.g.
        ``"Loading program: /programs/my_program.urp"``
        or an error message.
        """
        return self._send_cmd(f"load {program_path}")

    def play(self) -> str:
        """
        Start the currently loaded program.

        Returns ``"Starting program"`` on success or
        ``"Failed to execute: play"`` on failure.
        """
        return self._send_cmd("play")

    def stop(self) -> str:
        """
        Stop the currently running program.

        Returns ``"Stopped"`` on success or ``"Failed to execute: stop"``.
        """
        return self._send_cmd("stop")

    def pause(self) -> str:
        """
        Pause the currently running program.

        Returns ``"Pausing program"`` on success or
        ``"Failed to execute: pause"``.
        """
        return self._send_cmd("pause")

    def is_running(self) -> bool:
        """
        Return ``True`` if a program is currently executing.

        Wraps the ``running`` command (available since v1.6).
        """
        response = self._send_cmd("running")
        return response.lower() == "program running: true"

    def program_state(self) -> str:
        """
        Return the state of the active program.

        Possible values: ``"STOPPED"``, ``"PLAYING"``, ``"PAUSED"``
        (CB3/CB3.1 only for PAUSED). Also returns the path to the loaded
        program file when a program is loaded.

        Available since v1.8.
        """
        return self._send_cmd("programState")

    def is_program_saved(self) -> str:
        """
        Return the save state of the active program and its file path.

        Returns ``"True <path>"`` or ``"False <path>"``.

        Available since v1.8.
        """
        return self._send_cmd("isProgramSaved")

    def get_loaded_program(self) -> str:
        """
        Return which program is currently loaded.

        Returns ``"Loaded program: <path>"`` or ``"No program loaded"``.

        Available since v1.6.
        """
        return self._send_cmd("get loaded program")

    # ------------------------------------------------------------------
    # Installation  (available since v3.2)
    # ------------------------------------------------------------------

    def load_installation(self, installation_path: str) -> str:
        """
        Load an installation file.

        Blocks until loading completes or fails. Fails if the installation
        requires safety confirmation.

        Returns the server response string.
        """
        return self._send_cmd(f"load installation {installation_path}")

    # ------------------------------------------------------------------
    # Robot mode & safety mode  (available since v1.6 / v3.0)
    # ------------------------------------------------------------------

    def robot_mode(self) -> str:
        """
        Return the current robot mode.

        CB3 returns a string such as ``"Robotmode: RUNNING"``.
        Possible modes: ``NO_CONTROLLER``, ``DISCONNECTED``,
        ``CONFIRM_SAFETY``, ``BOOTING``, ``POWER_OFF``, ``POWER_ON``,
        ``IDLE``, ``BACKDRIVE``, ``RUNNING``.

        Available since v1.6.
        """
        return self._send_cmd("robotmode")

    def safety_mode(self) -> str:
        """
        Return the current safety mode.

        Returns ``"Safetymode: <MODE>"`` where MODE is one of:
        ``NORMAL``, ``REDUCED``, ``PROTECTIVE_STOP``, ``RECOVERY``,
        ``SAFEGUARD_STOP``, ``SYSTEM_EMERGENCY_STOP``,
        ``ROBOT_EMERGENCY_STOP``, ``VIOLATION``, ``FAULT``.

        Available since v3.0.
        """
        return self._send_cmd("safetymode")

    # ------------------------------------------------------------------
    # Power & brakes  (available since v3.0)
    # ------------------------------------------------------------------

    def power_on(self) -> str:
        """
        Power on the robot arm.

        Returns ``"Powering on"``.
        """
        return self._send_cmd("power on")

    def power_off(self) -> str:
        """
        Power off the robot arm.

        Returns ``"Powering off"``.
        """
        return self._send_cmd("power off")

    def brake_release(self) -> str:
        """
        Release the brakes on the robot arm.

        Returns ``"Brake releasing"``.
        """
        return self._send_cmd("brake release")

    # ------------------------------------------------------------------
    # Safety controls  (available since v3.1 / v3.7)
    # ------------------------------------------------------------------

    def unlock_protective_stop(self) -> str:
        """
        Close the current popup and unlock a protective stop.

        Fails if fewer than 5 seconds have passed since the stop occurred.
        Always inspect the cause before unlocking.

        Returns ``"Protective stop releasing"`` on success or an error
        message indicating the 5-second wait requirement.

        Available since v3.1.
        """
        return self._send_cmd("unlock protective stop")

    def close_safety_popup(self) -> str:
        """
        Close a safety popup.

        Returns ``"closing safety popup"``.

        Available since v3.1.
        """
        return self._send_cmd("close safety popup")

    def restart_safety(self) -> str:
        """
        Restart the safety system after a safety fault or violation.

        After reboot the robot will be in Power Off state. It is strongly
        recommended to check the error log before using this command.

        Returns ``"Restarting safety"``.

        Available since v3.7.
        """
        return self._send_cmd("restart safety")

    # ------------------------------------------------------------------
    # Popup UI  (available since v1.6)
    # ------------------------------------------------------------------

    def popup(self, text: str) -> str:
        """
        Display a popup on the Teach Pendant with the given text.

        The text is translated to the selected language if the key exists
        in the language file.

        Returns ``"showing popup"``.
        """
        return self._send_cmd(f"popup {text}")

    def close_popup(self) -> str:
        """
        Close the current popup.

        Returns ``"closing popup"``.
        """
        return self._send_cmd("close popup")

    # ------------------------------------------------------------------
    # User roles  (available since v1.8)
    # ------------------------------------------------------------------

    def set_user_role(self, role: str) -> str:
        """
        Set the user role on the Welcome screen.

        Parameters
        ----------
        role:
            One of ``"programmer"``, ``"operator"``, ``"none"``,
            ``"locked"``, or ``"restricted"`` (v3.1+).

            - ``programmer``: Setup buttons disabled; Expert Mode available.
            - ``operator``: Only RUN Program and SHUTDOWN Robot enabled;
              Expert Mode cannot be activated.
            - ``none``: All buttons enabled.
            - ``locked``: All buttons disabled.
            - ``restricted`` (v3.1+): Like operator but no Move tab access.

        Note: If the Welcome screen is not active the new role does not
        take effect until the user navigates to it.

        Returns ``"Setting user role: <role>"`` on success or
        ``"Failed setting user role: <role>"``.
        """
        return self._send_cmd(f"setUserRole {role}")

    def get_user_role(self) -> str:
        """
        Return the current user role.

        Returns one of: ``PROGRAMMER``, ``OPERATOR``, ``NONE``,
        ``LOCKED``, ``RESTRICTED``.
        """
        return self._send_cmd("getUserRole")

    # ------------------------------------------------------------------
    # Logging  (available since v1.8)
    # ------------------------------------------------------------------

    def add_to_log(self, message: str) -> str:
        """
        Add a message to the robot's log history.

        Returns ``"Added log message"`` or ``"No log message to add"``.
        """
        return self._send_cmd(f"addToLog {message}")

    # ------------------------------------------------------------------
    # System information  (available since v1.8 / v3.12)
    # ------------------------------------------------------------------

    def polyscope_version(self) -> str:
        """
        Return the Polyscope software version string, e.g. ``"3.0.15547"``.

        Available since v1.8.
        """
        return self._send_cmd("PolyscopeVersion")

    def get_serial_number(self) -> str:
        """
        Return the robot serial number, e.g. ``"2017351234"``.

        Available since v3.12.
        """
        return self._send_cmd("get serial number")

    def get_robot_model(self) -> str:
        """
        Return the robot model: ``UR3``, ``UR5``, or ``UR10``.

        Available since v3.12.
        """
        return self._send_cmd("get robot model")

    # ------------------------------------------------------------------
    # Diagnostics / flight reports  (available since v3.13)
    # ------------------------------------------------------------------

    def generate_flight_report(self, report_type: str = "system") -> str:
        """
        Trigger a Flight Report of the given type.

        Parameters
        ----------
        report_type:
            ``"controller"`` – information for diagnosing controller errors
            (protective stops, faults, violations).
            ``"software"``   – information for Polyscope software failures.
            ``"system"``     – robot configuration, programs, and
            installations (default).

        Note: You must wait at least 30 seconds between ``software`` or
        ``controller`` reports. This command can take several minutes to
        complete.

        Returns the report ID on success or an error message.

        Available since v3.13.
        """
        return self._send_cmd(f"generate flight report {report_type}")

    def generate_support_file(self, directory_path: str) -> str:
        """
        Generate a support archive and save it to *directory_path*.

        Generates a "system" flight report and creates a compressed ZIP of
        all existing flight reports. The result is named
        ``ur_<serial>_YYYY-MM-DD_HH-MM-SS.zip`` inside the given directory.

        *directory_path* must already exist inside the robot's programs
        directory (can point to ``usbdisk`` sub-folders).

        Returns ``"Completed successfully: <filename>"`` or an error
        message. Can take up to 10 minutes.

        Available since v3.13.
        """
        return self._send_cmd(f"generate support file {directory_path}")

    # ------------------------------------------------------------------
    # System shutdown  (available since v1.4)
    # ------------------------------------------------------------------

    def shutdown(self) -> str:
        """
        Shut down and power off the robot and controller.

        Returns ``"Shutting down"``.

        .. warning::
            This turns off the robot controller. Reconnection requires
            physical or remote power cycling.
        """
        return self._send_cmd("shutdown")

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def wait_for_program_to_finish(self, poll_interval: float = 1.0, timeout: float = 300.0) -> bool:
        """
        Block until the running program stops or *timeout* seconds elapse.

        Parameters
        ----------
        poll_interval:
            Seconds between ``running`` queries.
        timeout:
            Maximum seconds to wait before returning ``False``.

        Returns ``True`` if the program finished within *timeout*,
        ``False`` if the timeout was reached.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.is_running():
                return True
            time.sleep(poll_interval)
        return False

    def power_on_and_release_brakes(self, settle_time: float = 8.0) -> tuple[str, str]:
        """
        Send ``power on`` followed by ``brake release`` with a brief
        pause to allow the arm to initialise.

        Parameters
        ----------
        settle_time:
            Seconds to wait between power-on and brake release so the
            controller has time to reach POWER_ON mode.

        Returns a tuple of ``(power_on_response, brake_release_response)``.
        """
        power_resp = self.power_on()
        time.sleep(settle_time)
        brake_resp = self.brake_release()
        return power_resp, brake_resp
