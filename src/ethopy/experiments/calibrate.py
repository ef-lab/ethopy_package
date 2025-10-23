import logging
import time
import os
from importlib import import_module

import pygame

try:
    import pygame_menu
    IMPORT_PYGAME_MENU = True
except ImportError:
    IMPORT_PYGAME_MENU = False

log = logging.getLogger(__name__)


class Experiment:
    """Calibration experiment with Pi 5 compatibility
    
    Main menu where every time we want to move to new one, clean it
    and render the new components

    Menu order:
    1. pressure menu: define the air pressure in PSI
    The next menus run in loop for all the number of pulses
    2. a menu to check that pads has been placed under the ports
    3. run the pulses on every port
    4. port weight menu to define the weight in every port
    """

    def __init__(self):
        self.params = None
        self.logger = None
        self.sync = False
        self.cal_idx = 0
        self.msg = ""
        self.pulse = 0
        
        # Screen dimensions - will be auto-detected
        self.screen_width = 800
        self.screen_height = 480
        
        self.ports = None
        self.port = None
        self.interface = None
        self.screen = None
        self.menu = None
        self.theme = None
        
        # Pi 5 compatibility flags
        self.is_fullscreen = False
        self.display_scale = 1.0
        
        if not globals()["IMPORT_PYGAME_MENU"]:
            raise ImportError(
                "You need to install the pygame-menu: pip install pygame-menu"
            )

    def setup(self, logger, params):
        """Setup experiment with Pi 5 compatibility"""
        self.params = params
        self.logger = logger

        # Get interface configuration
        interface_module = self.logger.get(
            schema="interface",
            table="SetupConfiguration",
            fields=["interface"],
            key={"setup_conf_idx": self.params["setup_conf_idx"]},
        )[0]
        interface = getattr(
            import_module(f"ethopy.interfaces.{interface_module}"), interface_module
        )
        self.setup_conf_idx = self.params["setup_conf_idx"]

        # Initialize interface (this will use our Pi 5 compatible RPPorts)
        try:
            self.interface = interface(exp=self, callbacks=False)
            log.info("Interface initialized successfully")
        except Exception as e:
            log.error(f"Failed to initialize interface: {e}")
            raise

        # Initialize pygame with Pi 5 compatibility
        self._init_pygame()
        
        # Setup pygame menu theme
        self._setup_theme()
        
        # Create main menu
        self._create_main_menu()

        # Initialize calibration variables
        self.pressure = None
        self.curr = ""
        self.stop = False
        
        # Start with the pressure menu
        self.create_pressure_menu()
        
        # Run the experiment
        self.run()

    def _init_pygame(self):
        """Initialize pygame with Pi 5 compatibility"""
        if not pygame.get_init():
            pygame.init()
            
        # Auto-detect screen resolution for Pi 5
        try:
            info = pygame.display.Info()
            detected_width = info.current_w
            detected_height = info.current_h
            
            log.info(f"Detected screen resolution: {detected_width}x{detected_height}")
            
            # Use detected resolution if reasonable, otherwise fallback
            if detected_width > 400 and detected_height > 300:
                self.screen_width = detected_width
                self.screen_height = detected_height
            else:
                log.warning("Using fallback resolution 800x480")
                
        except Exception as e:
            log.warning(f"Could not detect screen resolution: {e}")

        # Calculate scaling factor for UI elements
        self.display_scale = min(self.screen_width / 800, self.screen_height / 480)
        
        # Set display mode with Pi 5 compatibility
        try:
            if self.logger.is_pi:
                # Try fullscreen mode first
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height), 
                    pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE
                )
                self.is_fullscreen = True
                log.info("Fullscreen mode activated")
                
                # Hide mouse cursor for kiosk mode
                pygame.mouse.set_visible(False)
                
            else:
                # Windowed mode for development
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                log.info("Windowed mode activated")
                
        except pygame.error as e:
            log.error(f"Failed to set display mode: {e}")
            # Fallback to windowed mode
            self.screen = pygame.display.set_mode((800, 480))
            self.screen_width = 800
            self.screen_height = 480
            self.display_scale = 1.0
            log.info("Fallback to 800x480 windowed mode")

        pygame.display.set_caption("EthoPy Calibration")

    def _setup_theme(self):
        """Setup pygame menu theme with scaling"""
        self.theme = pygame_menu.themes.THEME_DARK.copy()
        self.theme.background_color = (0, 0, 0)
        self.theme.title_background_color = (43, 43, 43)
        
        # Scale font sizes based on display
        self.theme.title_font_size = int(35 * self.display_scale)
        self.theme.widget_font_size = int(30 * self.display_scale)
        
        self.theme.widget_alignment = pygame_menu.locals.ALIGN_CENTER
        self.theme.widget_font_color = (255, 255, 255)
        self.theme.widget_padding = 0

    def _create_main_menu(self):
        """Create the main menu with proper dimensions"""
        self.menu = pygame_menu.Menu(
            "",
            self.screen_width,
            self.screen_height,
            center_content=False,
            mouse_motion_selection=True,
            onclose=None,
            overflow=False,
            theme=self.theme,
        )

    def run(self) -> None:
        """Calibration mainloop with improved error handling"""
        clock = pygame.time.Clock()  # Add FPS control
        
        try:
            while not self.stop:
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.QUIT:
                        self.stop = True
                        break
                    # Add escape key to exit fullscreen/application
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.stop = True
                            break
                        elif event.key == pygame.K_F11 and not self.logger.is_pi:
                            # Toggle fullscreen in development mode
                            self._toggle_fullscreen()

                if self.menu and self.menu.is_enabled() and not self.stop:
                    try:
                        self.menu.update(events)
                        # Clear screen before drawing
                        self.screen.fill((0, 0, 0))
                        self.menu.draw(self.screen)
                        pygame.display.flip()  # Use flip() for better performance
                    except pygame.error as e:
                        log.error(f"Display error: {e}")
                        break
                        
                # Limit FPS to reduce CPU usage
                clock.tick(60)
                
        except KeyboardInterrupt:
            log.info("Keyboard interrupt received")
        except Exception as e:
            log.error(f"Unexpected error in main loop: {e}")
        finally:
            self.cleanup()

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode (development only)"""
        try:
            if self.is_fullscreen:
                self.screen = pygame.display.set_mode((800, 480))
                self.is_fullscreen = False
                pygame.mouse.set_visible(True)
            else:
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height), pygame.FULLSCREEN
                )
                self.is_fullscreen = True
                pygame.mouse.set_visible(False)
        except Exception as e:
            log.error(f"Failed to toggle fullscreen: {e}")

    def cleanup(self):
        """Cleanup pygame and interface resources with better error handling"""
        log.info("Starting cleanup...")
        
        # Cleanup pygame
        if pygame.get_init():
            try:
                pygame.event.clear()
                
                if hasattr(self, "menu") and self.menu:
                    self.menu.disable()
                    
                pygame.mouse.set_visible(True)  # Show cursor before exit
                pygame.display.quit()
                pygame.quit()
                log.info("Pygame cleaned up successfully")
            except Exception as e:
                log.warning(f"Error during pygame cleanup: {e}")

        # Cleanup interface
        if hasattr(self, "interface") and self.interface:
            try:
                self.interface.cleanup()
                log.info("Interface cleaned up successfully")
            except Exception as e:
                log.warning(f"Error during interface cleanup: {e}")

        # Update logger status
        if hasattr(self, "logger") and self.logger:
            try:
                self.logger.update_setup_info({"status": "ready"})
                log.info("Logger status updated")
            except Exception as e:
                log.warning(f"Error updating logger status: {e}")

    def exit(self):
        """Exit function after the Experiment has finished"""
        try:
            if self.menu:
                self.menu.clear()
                # Scale exit message font
                exit_font_size = int(30 * self.display_scale)
                self.menu.add.label(
                    "Done calibrating!!", 
                    float=True, 
                    font_size=exit_font_size
                ).translate(int(20 * self.display_scale), int(80 * self.display_scale))
                
                try:
                    self.screen.fill((0, 0, 0))
                    self.menu.draw(self.screen)
                    pygame.display.flip()
                    time.sleep(2)
                except pygame.error:
                    pass  # Display might already be quit

            self.stop = True
        except Exception as e:
            log.warning(f"Error during exit: {e}")
            self.stop = True

    def create_pressure_menu(self):
        """The First menu in Calibration where air pressure in PSI is defined"""
        self.menu.clear()
        self.button_input("Enter air pressure (PSI)", self.create_pulsenum_menu)

    def create_pulsenum_menu(self):
        """Before start pulses show a label in screen to place a pad under the ports"""
        self.pressure = self.curr
        self.curr = ""
        if self.cal_idx < len(self.params["pulsenum"]):
            self.menu.clear()
            
            # Scale UI elements
            label_font_size = int(30 * self.display_scale)
            button_font_size = int(30 * self.display_scale)
            
            self.menu.add.label(
                "Place zero-weighted pad under the port", 
                float=True, 
                font_size=label_font_size
            ).translate(int(20 * self.display_scale), int(80 * self.display_scale))
            
            self.menu.add.button(
                "OK",
                self.create_pulse_num,
                align=pygame_menu.locals.ALIGN_LEFT,
                float=True,
                padding=(10, 10, 10, 10),
                background_color=(0, 128, 0),
                font_size=button_font_size,
            ).translate(int(400 * self.display_scale), int(140 * self.display_scale))
        else:
            self.exit()

    def create_pulse_num(self):
        """Display the pulses"""
        self.pulse = 0
        msg = f"Pulse {self.pulse + 1}/{self.params['pulsenum'][self.cal_idx]}"
        self.menu.clear()
        
        # Scale pulse label
        pulse_font_size = int(40 * self.display_scale)
        
        pulses_label = self.menu.add.label(
            msg,
            float=True,
            label_id="pulses_label",
            font_size=pulse_font_size,
            background_color=(0, 15, 15),
        ).translate(0, int(50 * self.display_scale))
        
        # Adds a function to the Widget to be executed each time the label is drawn
        pulses_label.add_draw_callback(self.run_pulses)

    def run_pulses(self, widget, menu):
        """This function is executed each time the label is drawn"""
        if self.pulse < self.params["pulsenum"][self.cal_idx]:
            self.msg = f"Pulse {self.pulse + 1}/{self.params['pulsenum'][self.cal_idx]}"
            log.info(f"\r{self.msg}")
            widget.set_title(self.msg)
            
            for port in self.params["ports"]:
                try:
                    self.interface.give_liquid(
                        port, self.params["duration"][self.cal_idx]
                    )
                except Exception as error:
                    log.error(f"Calibration Error: {error}")
                    self.exit()
                    return

                time.sleep(
                    self.params["duration"][self.cal_idx] / 1000
                    + self.params["pulse_interval"][self.cal_idx] / 1000
                )
            self.pulse += 1
        else:
            self.cal_idx += 1
            self.ports = self.params["ports"].copy()
            self.create_port_weight()

    def create_port_weight(self):
        """A menu with numpad for defining the water weight in every port"""
        self.menu.clear()
        cal_idx = self.cal_idx - 1
        
        if self.params["save"]:
            if len(self.ports) != 0:
                if len(self.ports) != len(self.params["ports"]):
                    self.log_pulse_weight(
                        self.params["duration"][cal_idx],
                        self.port,
                        self.params["pulsenum"][cal_idx],
                        self.curr,
                        self.pressure,
                    )

                self.port = self.ports.pop(0)
                self.button_input(
                    f"Enter weight for port {self.port}", self.create_port_weight
                )
            else:
                self.log_pulse_weight(
                    self.params["duration"][cal_idx],
                    self.port,
                    self.params["pulsenum"][cal_idx],
                    self.curr,
                    self.pressure,
                )
                self.create_pulsenum_menu()
        else:
            self.create_pulsenum_menu()

    def button_input(self, message: str, _func):
        """Create a label with a numpad"""
        label_font_size = int(25 * self.display_scale)
        self.menu.add.label(message, font_size=label_font_size)
        self.num_pad(_func)

    def num_pad(self, _func):
        """Create numpad with scaling for different screen sizes"""
        self.num_pad_disp = self.menu.add.label(
            "",
            background_color=None,
            margin=(10, 0),
            selectable=True,
            selection_effect=None,
        )
        self.menu.add.vertical_margin(int(10 * self.display_scale))
        cursor = pygame_menu.locals.CURSOR_HAND
        self.curr = ""

        # Scale button dimensions
        frame_width = int(299 * self.display_scale)
        frame_height = int(54 * self.display_scale)
        button_width = int(74 * self.display_scale)
        button_height = int(54 * self.display_scale)

        # Add horizontal frames with scaled dimensions
        f1 = self.menu.add.frame_h(frame_width, frame_height, margin=(0, 0))
        b1 = f1.pack(self.menu.add.button("1", lambda: self._press(1), cursor=cursor))
        b2 = f1.pack(
            self.menu.add.button("2", lambda: self._press(2), cursor=cursor),
            align=pygame_menu.locals.ALIGN_CENTER,
        )
        b3 = f1.pack(
            self.menu.add.button("3", lambda: self._press(3), cursor=cursor),
            align=pygame_menu.locals.ALIGN_RIGHT,
        )
        self.menu.add.vertical_margin(int(5 * self.display_scale))

        f2 = self.menu.add.frame_h(frame_width, frame_height, margin=(0, 0))
        b4 = f2.pack(self.menu.add.button("4", lambda: self._press(4), cursor=cursor))
        b5 = f2.pack(
            self.menu.add.button("5", lambda: self._press(5), cursor=cursor),
            align=pygame_menu.locals.ALIGN_CENTER,
        )
        b6 = f2.pack(
            self.menu.add.button("6", lambda: self._press(6), cursor=cursor),
            align=pygame_menu.locals.ALIGN_RIGHT,
        )
        self.menu.add.vertical_margin(int(5 * self.display_scale))

        f3 = self.menu.add.frame_h(frame_width, frame_height, margin=(0, 0))
        b7 = f3.pack(self.menu.add.button("7", lambda: self._press(7), cursor=cursor))
        b8 = f3.pack(
            self.menu.add.button("8", lambda: self._press(8), cursor=cursor),
            align=pygame_menu.locals.ALIGN_CENTER,
        )
        b9 = f3.pack(
            self.menu.add.button("9", lambda: self._press(9), cursor=cursor),
            align=pygame_menu.locals.ALIGN_RIGHT,
        )
        self.menu.add.vertical_margin(int(5 * self.display_scale))

        f4 = self.menu.add.frame_h(frame_width, frame_height, margin=(0, 0))
        delete = f4.pack(
            self.menu.add.button("<", lambda: self._press("<"), cursor=cursor),
            align=pygame_menu.locals.ALIGN_RIGHT,
        )
        b0 = f4.pack(
            self.menu.add.button("0", lambda: self._press(0), cursor=cursor),
            align=pygame_menu.locals.ALIGN_CENTER,
        )
        dot = f4.pack(
            self.menu.add.button(" .", lambda: self._press("."), cursor=cursor),
            align=pygame_menu.locals.ALIGN_LEFT,
        )
        self.menu.add.vertical_margin(int(5 * self.display_scale))

        f5 = self.menu.add.frame_h(frame_width, frame_height, margin=(0, 0))
        ok = f5.pack(
            self.menu.add.button("OK", lambda: self._press("ok", _func), cursor=cursor),
            align=pygame_menu.locals.ALIGN_CENTER,
        )

        # Add decorator for each object with scaled dimensions
        rect_offset_x = int(-37 * self.display_scale)
        rect_offset_y = int(-27 * self.display_scale)
        rect_width = int(button_width)
        rect_height = int(button_height)
        
        for widget in (b1, b2, b3, b4, b5, b6, b7, b8, b9, b0, ok, delete, dot):
            w_deco = widget.get_decorator()
            if widget != ok:
                w_deco.add_rectangle(rect_offset_x, rect_offset_y, rect_width, rect_height, (15, 15, 15))
                on_layer = w_deco.add_rectangle(rect_offset_x, rect_offset_y, rect_width, rect_height, (84, 84, 84))
            else:
                w_deco.add_rectangle(rect_offset_x, rect_offset_y, rect_width, rect_height, (0, 128, 0))
                on_layer = w_deco.add_rectangle(rect_offset_x, rect_offset_y, rect_width, rect_height, (40, 171, 187))
            w_deco.disable(on_layer)
            widget.set_attribute("on_layer", on_layer)

            def widget_select(sel: bool, wid: "pygame_menu.widgets.Widget", _):
                """Function triggered if widget is selected"""
                lay = wid.get_attribute("on_layer")
                if sel:
                    wid.get_decorator().enable(lay)
                else:
                    wid.get_decorator().disable(lay)

            widget.set_onselect(widget_select)
            widget.set_padding((2, 19, 0, 23))

    def _press(self, digit, _func=None) -> None:
        """Press numpad digit"""
        if digit == "ok":
            if self.curr != "":
                _func()
        elif digit == "<":
            self.curr = ""
            self.num_pad_disp.set_title("")
        else:
            if len(self.curr) <= 9:
                self.curr += str(digit)
            self.num_pad_disp.set_title(self.curr)

    def log_pulse_weight(self, pulse_dur, port, pulse_num, weight=0, pressure=0):
        """Log calibration data to database"""
        try:
            key = dict(
                setup=self.logger.setup, 
                port=port, 
                date=time.strftime("%Y-%m-%d")
            )
            
            self.logger.put(
                table="PortCalibration",
                tuple=key,
                schema="interface",
                priority=5,
                ignore_extra_fields=True,
                validate=True,
                block=True,
                replace=False,
            )
            
            self.logger.put(
                table="PortCalibration.Liquid",
                schema="interface",
                replace=True,
                ignore_extra_fields=True,
                tuple=dict(
                    key,
                    pulse_dur=pulse_dur,
                    pulse_num=pulse_num,
                    weight=weight,
                    pressure=pressure,
                ),
            )
            log.info(f"Logged calibration data for port {port}")
            
        except Exception as e:
            log.error(f"Failed to log calibration data: {e}")