"""EyeLink x PsychoPy for Stanford CNI
@Author: Shawn T. Schwartz 
@Email: stschwartz@stanford.edu
@Date: 2/17/2023
@Links: https://shawnschwartz.com
"""
import sys, os, random, glob, math, csv, uuid, errno, json, pickle, time, pylink, platform
import numpy as np
import pandas as pd
from psychopy import visual, core, event, monitors, tools, data, gui, logging
from EyeLinkCoreGraphicsPsychoPy.EyeLinkCoreGraphicsPsychoPy.EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy

"""Global PsychoPy Settings
"""
logging.console.setLevel(logging.CRITICAL) # show only critical messages in the PsychoPy console

def convert_color(color):
    return [round(((n/127.5)-1), 2) for n in color]

class EyelinkCNI():
    def __init__(self, *args, **kwargs):
        super(EyelinkCNI, self).__init__(*args, **kwargs)
        
        self.color_text = convert_color((0, 0, 0))
        self.color_bg = convert_color((128, 128, 128))
        self.color_eyelink_bg = convert_color((128, 128, 128))
        
        self.msg_calib_el = "<<< We will now calibrate the eye-tracker >>>"
        
        self.monitor_name = 'Experiment Monitor'
        self.monitor_width = 53
        self.monitor_distance = 70
        self.monitor_px = [1920, 1080]
        self.fullscreen = True
        
        self.eyetrack = False
        self.eyelink_ip = '100.1.1.1'
        
        self.subject_id = 'demo' # customize your way to store this open task init
        
        self.experiment_monitor = monitors.Monitor(
            self.monitor_name, 
            width = self.monitor_width,
            distance = self.monitor_distance)
        self.experiment_monitor.setSizePix(self.monitor_px)
        
        vars(self).update(kwargs)
        
    def open_window(self, **kwargs):
        """Opens the psychopy window.
        """
        
        self.experiment_window = visual.Window(
            monitor = self.experiment_monitor, 
            fullscr = self.fullscreen, 
            color = self.color_bg, 
            winType = 'pyglet',
            colorSpace = 'rgb', 
            units = 'pix',
            allowGUI = False, 
            **kwargs)
        
    def make_message(self, message_string, key_list=['space']):
        message = visual.TextStim(win=self.experiment_window, colorSpace='rgb255', color=self.color_text, text=message_string)
        message.draw()
        self.experiment_window.flip()
        keys = event.waitKeys(keyList=key_list)
        return keys
    
    """Eyelink helper functions
    """
    def _swap_bg_color_to_calibration_screen(self):
        self.experiment_window.color = self.color_eyelink_bg
        self.experiment_window.flip()

    def _swap_bg_color_to_task_screen(self):
        self.experiment_window.color = self.color_bg
        self.experiment_window.flip()

    def _connect_eyelink(self):
        try:
            self.el_tracker = pylink.EyeLink(self.eyelink_ip)
        except RuntimeError as error:
            print('ERROR:', error)
            self.experiment_window.close()
            sys.exit(1)

    def _open_edf_file(self, phase, block):
        # IMPORTANT: limit to 8 characters or else eyelink will crash when connecting to host
        # customize for your application (i.e., passing in something like self.subject_id, etc.)
        self.edf_file = str(self.subject_id + phase[0:1] + str(block) + '.EDF')

        try:
            self.el_tracker.openDataFile(self.edf_file)
        except RuntimeError as err:
            print('ERROR:', err)
            # close the link if one is open
            if self.el_tracker.isConnected():
                self.el_tracker.close()
            self.experiment_window.close()
            sys.exit(1)

    def _send_edf_preamble(self):
        preamble_text = 'RECORDED BY %s' % os.path.basename(__file__)
        self.el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)

    def _config_eyelink(self):
        self.el_tracker.setOfflineMode() # put the tracker in offline mode before tracking parameters are changed
        eyelink_ver = 0
        vstr = self.el_tracker.getTrackerVersionString()
        eyelink_ver = int(vstr.split()[-1].split('.')[0])
        print('Running experiment on %s, version %d' % (vstr, eyelink_ver))

        file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
        link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'

        if eyelink_ver > 3:
            file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
            link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
        else:
            file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
            link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
        self.el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
        self.el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
        self.el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
        self.el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)

        if eyelink_ver > 2:
            self.el_tracker.sendCommand("sample_rate 1000")

        self.el_tracker.sendCommand("calibration_type = HV9")
        self.el_tracker.sendCommand("randomize_calibration_order = NO")
        self.el_tracker.sendCommand("calibration_area_proportion 1.0 1.0")
        self.el_tracker.sendCommand("validation_area_proportion 0.6 0.6")

        self.el_tracker.sendCommand("button_function 5 'accept_target_fixation'")

        # get the native screen resolution used by psychopy
        scn_width = int(self.experiment_window.size[0])
        scn_height = int(self.experiment_window.size[1])

        # pass the display pixel coordinates (left, top, right, bottom) to the tracker
        el_coords = "screen_pixel_coords = 0.0, 0.0, %d, %d" % (scn_width, scn_height)
        self.el_tracker.sendCommand(el_coords)

        # write a DISPLAY_COORDS message to the EDF file
        dv_coords = "DISPLAY_COORDS 0 0 %d %d" % (scn_width, scn_height)
        self.el_tracker.sendMessage(dv_coords)

        calib_x0 = 400
        calib_x1 = 960
        calib_x2 = 1420
        calib_y0 = 300
        calib_y1 = 540
        calib_y2 = 780

        calib_positions = "calibration_targets = %d,%d %d,%d %d,%d %d,%d %d,%d %d,%d %d,%d %d,%d %d,%d" % (
            calib_x1, calib_y1, 
            calib_x1, calib_y0, 
            calib_x1, calib_y2, 
            calib_x0, calib_y1, 
            calib_x2, calib_y1, 
            calib_x0, calib_y0, 
            calib_x2, calib_y0, 
            calib_x0, calib_y2,
            calib_x2, calib_y2)
        self.el_tracker.sendCommand(calib_positions)

    def _calibrate_eyelink(self):
        self._swap_bg_color_to_calibration_screen()
        self.make_message(self.msg_calib_el)
        genv = EyeLinkCoreGraphicsPsychoPy(self.el_tracker, self.experiment_window)
        pylink.openGraphicsEx(genv)
        self.el_tracker.doTrackerSetup()

    def _start_eyelink_recording(self):
        self.el_tracker.startRecording(1, 1, 1, 1)
        time.sleep(.1)  # required
        self.el_tracker.sendMessage('start_run')
        self._swap_bg_color_to_task_screen()

    def trigger_eyelink(self, phase, block):
        self._connect_eyelink()
        self._open_edf_file(phase, block)
        self._send_edf_preamble()
        self._config_eyelink()
        self._calibrate_eyelink()
        self._start_eyelink_recording()

    def disconnect_eyelink(self):
        self.el_tracker.sendMessage('end_run')
        if self.el_tracker.isConnected():
            time.sleep(.1)  # required
            self.el_tracker.stopRecording()
            self.el_tracker.setOfflineMode() # put the tracker into offline mode
            self.el_tracker.sendCommand('clear_screen 0') # clear the host pc screen and wait for 500 ms
            pylink.msecDelay(500)
            self.el_tracker.closeDataFile() # close the EDF data file on the host pc
            pylink.closeGraphics()
            self.el_tracker.close() # disconnect the tracker
