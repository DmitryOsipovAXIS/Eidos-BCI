#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on May 27, 2026, at 09:50
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'Flickering'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1664, 1109]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Code\\School\\3de\\Sem 2\\Bachelor Project\\AIandData\\bci-ssvep\\src\\collection\\collecting_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Start" ---
    title_start = visual.TextStim(win=win, name='title_start',
        text='',
        font='Arial',
        pos=(0, 0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    ready_to_start = visual.TextStim(win=win, name='ready_to_start',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Countdown" ---
    title_countdown = visual.TextStim(win=win, name='title_countdown',
        text='',
        font='Arial',
        pos=(0, 0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    three = visual.TextStim(win=win, name='three',
        text='3',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    two = visual.TextStim(win=win, name='two',
        text='2',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    one = visual.TextStim(win=win, name='one',
        text='1',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "Collect" ---
    title = visual.TextStim(win=win, name='title',
        text='',
        font='Arial',
        pos=(0, 0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    stimuli = visual.Rect(
        win=win, name='stimuli',
        width=(0.9, 0.7)[0], height=(0.9, 0.7)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    # Run 'Begin Experiment' code from stimuli_code
    import sys
    import csv
    import numpy as np
    from datetime import datetime
    from pathlib import Path
    import time
    
    
    src_dir = str(Path(_thisDir).parents[0])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    no_eeg = '--no-eeg' in sys.argv
    serial_port = 'COM4'
    num_channels = 4
    
    for arg in sys.argv:
        if arg.startswith('--serial-port='):
            serial_port = arg.split('=')[1]
        if arg.startswith('--num-channels='):
            num_channels = int(arg.split('=')[1])
    
    stream = None
    if not no_eeg:
        from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
        from acquisition.brainflow_stream import BrainFlowStream
        BoardShim.disable_board_logger()
        p = BrainFlowInputParams()
        p.serial_port = serial_port
        stream = BrainFlowStream(board_id=BoardIds.NEUROPAWN_KNIGHT_BOARD.value, params=p, num_channels=num_channels)
        stream.prepare_session()
        stream.start_stream()
        time.sleep(2)
        for ch in range(1, 9):
            time.sleep(0.5)
            stream.board.config_board(f"chon_{ch}_12")
            time.sleep(1)
            stream.board.config_board(f"rldadd_{ch}")
            time.sleep(0.5)
        print("Board ready.")
    
    X_list = []
    y_list = []
    
    freqLeft = 7.5
    freqRight = 12
    
    with open('conditions.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    
    has_both_sides = any(r['side'] == 'LEFT' for r in rows) and any(r['side'] == 'RIGHT' for r in rows)
    
    for arg in sys.argv:
        if arg.startswith('--freq-left='):
            freqLeft = float(arg.split('=')[1])
        if arg.startswith('--freq-right='):
            freqRight = float(arg.split('=')[1])
    
    frame_rate = win.getActualFrameRate()
    max_freq = frame_rate / 2
    
    if freqLeft >= max_freq or freqRight >= max_freq:
        raise ValueError(f"Frequencies must be below {max_freq}Hz (half your frame rate of {frame_rate}Hz)")
    
    frames_per_half_cycle_left = int(frame_rate / (2 * freqLeft))
    frames_per_half_cycle_right = int(frame_rate / (2 * freqRight))
    frame_count = 0
    
    # --- Initialize components for Routine "Waiting" ---
    waiting_title = visual.TextStim(win=win, name='waiting_title',
        text='',
        font='Arial',
        pos=(0, 0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from waiting_code
    import csv
    
    with open('conditions.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    
    has_both_sides = any(r['side'] == 'LEFT' for r in rows) and any(r['side'] == 'RIGHT' for r in rows)
    
    # --- Initialize components for Routine "Done" ---
    done_title = visual.TextStim(win=win, name='done_title',
        text='Finished, you can relax now',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Start" ---
    # create an object to store info about Routine Start
    Start = data.Routine(
        name='Start',
        components=[title_start, ready_to_start],
    )
    Start.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Start
    Start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Start.tStart = globalClock.getTime(format='float')
    Start.status = STARTED
    thisExp.addData('Start.started', Start.tStart)
    Start.maxDuration = None
    # keep track of which components have finished
    StartComponents = Start.components
    for thisComponent in Start.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Start" ---
    Start.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 7.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *title_start* updates
        
        # if title_start is starting this frame...
        if title_start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            title_start.frameNStart = frameN  # exact frame index
            title_start.tStart = t  # local t and not account for scr refresh
            title_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(title_start, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'title_start.started')
            # update status
            title_start.status = STARTED
            title_start.setAutoDraw(True)
        
        # if title_start is active this frame...
        if title_start.status == STARTED:
            # update params
            title_start.setText('Welcome to training\n', log=False)
        
        # if title_start is stopping this frame...
        if title_start.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > title_start.tStartRefresh + 7-frameTolerance:
                # keep track of stop time/frame for later
                title_start.tStop = t  # not accounting for scr refresh
                title_start.tStopRefresh = tThisFlipGlobal  # on global time
                title_start.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'title_start.stopped')
                # update status
                title_start.status = FINISHED
                title_start.setAutoDraw(False)
        
        # *ready_to_start* updates
        
        # if ready_to_start is starting this frame...
        if ready_to_start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            ready_to_start.frameNStart = frameN  # exact frame index
            ready_to_start.tStart = t  # local t and not account for scr refresh
            ready_to_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(ready_to_start, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'ready_to_start.started')
            # update status
            ready_to_start.status = STARTED
            ready_to_start.setAutoDraw(True)
        
        # if ready_to_start is active this frame...
        if ready_to_start.status == STARTED:
            # update params
            ready_to_start.setText('Make sure you are ready to start\n', log=False)
        
        # if ready_to_start is stopping this frame...
        if ready_to_start.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > ready_to_start.tStartRefresh + 7-frameTolerance:
                # keep track of stop time/frame for later
                ready_to_start.tStop = t  # not accounting for scr refresh
                ready_to_start.tStopRefresh = tThisFlipGlobal  # on global time
                ready_to_start.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ready_to_start.stopped')
                # update status
                ready_to_start.status = FINISHED
                ready_to_start.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Start,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Start.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Start.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Start" ---
    for thisComponent in Start.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Start
    Start.tStop = globalClock.getTime(format='float')
    Start.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Start.stopped', Start.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if Start.maxDurationReached:
        routineTimer.addTime(-Start.maxDuration)
    elif Start.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-7.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "Countdown" ---
    # create an object to store info about Routine Countdown
    Countdown = data.Routine(
        name='Countdown',
        components=[title_countdown, three, two, one],
    )
    Countdown.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Countdown
    Countdown.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Countdown.tStart = globalClock.getTime(format='float')
    Countdown.status = STARTED
    thisExp.addData('Countdown.started', Countdown.tStart)
    Countdown.maxDuration = None
    # keep track of which components have finished
    CountdownComponents = Countdown.components
    for thisComponent in Countdown.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Countdown" ---
    Countdown.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *title_countdown* updates
        
        # if title_countdown is starting this frame...
        if title_countdown.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            title_countdown.frameNStart = frameN  # exact frame index
            title_countdown.tStart = t  # local t and not account for scr refresh
            title_countdown.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(title_countdown, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'title_countdown.started')
            # update status
            title_countdown.status = STARTED
            title_countdown.setAutoDraw(True)
        
        # if title_countdown is active this frame...
        if title_countdown.status == STARTED:
            # update params
            title_countdown.setText('Welcome to training\n', log=False)
        
        # if title_countdown is stopping this frame...
        if title_countdown.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > title_countdown.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                title_countdown.tStop = t  # not accounting for scr refresh
                title_countdown.tStopRefresh = tThisFlipGlobal  # on global time
                title_countdown.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'title_countdown.stopped')
                # update status
                title_countdown.status = FINISHED
                title_countdown.setAutoDraw(False)
        
        # *three* updates
        
        # if three is starting this frame...
        if three.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            three.frameNStart = frameN  # exact frame index
            three.tStart = t  # local t and not account for scr refresh
            three.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(three, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'three.started')
            # update status
            three.status = STARTED
            three.setAutoDraw(True)
        
        # if three is active this frame...
        if three.status == STARTED:
            # update params
            pass
        
        # if three is stopping this frame...
        if three.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > three.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                three.tStop = t  # not accounting for scr refresh
                three.tStopRefresh = tThisFlipGlobal  # on global time
                three.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'three.stopped')
                # update status
                three.status = FINISHED
                three.setAutoDraw(False)
        
        # *two* updates
        
        # if two is starting this frame...
        if two.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
            # keep track of start time/frame for later
            two.frameNStart = frameN  # exact frame index
            two.tStart = t  # local t and not account for scr refresh
            two.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(two, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'two.started')
            # update status
            two.status = STARTED
            two.setAutoDraw(True)
        
        # if two is active this frame...
        if two.status == STARTED:
            # update params
            pass
        
        # if two is stopping this frame...
        if two.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 2.0-frameTolerance:
                # keep track of stop time/frame for later
                two.tStop = t  # not accounting for scr refresh
                two.tStopRefresh = tThisFlipGlobal  # on global time
                two.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'two.stopped')
                # update status
                two.status = FINISHED
                two.setAutoDraw(False)
        
        # *one* updates
        
        # if one is starting this frame...
        if one.status == NOT_STARTED and tThisFlip >= 2.0-frameTolerance:
            # keep track of start time/frame for later
            one.frameNStart = frameN  # exact frame index
            one.tStart = t  # local t and not account for scr refresh
            one.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(one, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'one.started')
            # update status
            one.status = STARTED
            one.setAutoDraw(True)
        
        # if one is active this frame...
        if one.status == STARTED:
            # update params
            pass
        
        # if one is stopping this frame...
        if one.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 3.0-frameTolerance:
                # keep track of stop time/frame for later
                one.tStop = t  # not accounting for scr refresh
                one.tStopRefresh = tThisFlipGlobal  # on global time
                one.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'one.stopped')
                # update status
                one.status = FINISHED
                one.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Countdown,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Countdown.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Countdown.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Countdown" ---
    for thisComponent in Countdown.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Countdown
    Countdown.tStop = globalClock.getTime(format='float')
    Countdown.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Countdown.stopped', Countdown.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if Countdown.maxDurationReached:
        routineTimer.addTime(-Countdown.maxDuration)
    elif Countdown.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('conditions.csv'), 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        trials.status = STARTED
        if hasattr(thisTrial, 'status'):
            thisTrial.status = STARTED
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "Collect" ---
        # create an object to store info about Routine Collect
        Collect = data.Routine(
            name='Collect',
            components=[title, stimuli],
        )
        Collect.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from stimuli_code
        frame_count = 0
        if stream is not None:
            stream.board.get_board_data()
        # store start times for Collect
        Collect.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Collect.tStart = globalClock.getTime(format='float')
        Collect.status = STARTED
        thisExp.addData('Collect.started', Collect.tStart)
        Collect.maxDuration = None
        # keep track of which components have finished
        CollectComponents = Collect.components
        for thisComponent in Collect.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Collect" ---
        Collect.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 30.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *title* updates
            
            # if title is starting this frame...
            if title.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                title.frameNStart = frameN  # exact frame index
                title.tStart = t  # local t and not account for scr refresh
                title.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(title, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'title.started')
                # update status
                title.status = STARTED
                title.setAutoDraw(True)
            
            # if title is active this frame...
            if title.status == STARTED:
                # update params
                title.setText('Right', log=False)
            
            # if title is stopping this frame...
            if title.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > title.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    title.tStop = t  # not accounting for scr refresh
                    title.tStopRefresh = tThisFlipGlobal  # on global time
                    title.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'title.stopped')
                    # update status
                    title.status = FINISHED
                    title.setAutoDraw(False)
            
            # *stimuli* updates
            
            # if stimuli is starting this frame...
            if stimuli.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                stimuli.frameNStart = frameN  # exact frame index
                stimuli.tStart = t  # local t and not account for scr refresh
                stimuli.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(stimuli, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'stimuli.started')
                # update status
                stimuli.status = STARTED
                stimuli.setAutoDraw(True)
            
            # if stimuli is active this frame...
            if stimuli.status == STARTED:
                # update params
                pass
            
            # if stimuli is stopping this frame...
            if stimuli.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > stimuli.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    stimuli.tStop = t  # not accounting for scr refresh
                    stimuli.tStopRefresh = tThisFlipGlobal  # on global time
                    stimuli.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'stimuli.stopped')
                    # update status
                    stimuli.status = FINISHED
                    stimuli.setAutoDraw(False)
            # Run 'Each Frame' code from stimuli_code
            total = trials.nTotal
            current = trials.thisN + 1
            
            if has_both_sides:
                total_display = total // 2
                current_display = (trials.thisN // 2) + 1
            else:
                total_display = total
                current_display = current
            
            title.text = f"Trial {current_display} / {total_display} - {side}"
            
            if side == 'LEFT':
                cycle_position = frame_count % (frames_per_half_cycle_left * 2)
                stimuli.opacity = 1 if cycle_position < frames_per_half_cycle_left else 0
            else:
                cycle_position = frame_count % (frames_per_half_cycle_right * 2)
                stimuli.opacity = 1 if cycle_position < frames_per_half_cycle_right else 0
            
            frame_count += 1
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Collect,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Collect.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Collect.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Collect" ---
        for thisComponent in Collect.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Collect
        Collect.tStop = globalClock.getTime(format='float')
        Collect.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Collect.stopped', Collect.tStop)
        # Run 'End Routine' code from stimuli_code
        if stream is not None:
            eeg = stream.board.get_board_data()
            eeg_channels = eeg[:num_channels, :]
            label = 0 if side == 'LEFT' else 1
            X_list.append(eeg_channels)
            y_list.append(label)
            print(f"Trial {trials.thisN + 1} {side} — shape: {eeg_channels.shape}")
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Collect.maxDurationReached:
            routineTimer.addTime(-Collect.maxDuration)
        elif Collect.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-30.000000)
        
        # --- Prepare to start Routine "Waiting" ---
        # create an object to store info about Routine Waiting
        Waiting = data.Routine(
            name='Waiting',
            components=[waiting_title],
        )
        Waiting.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for Waiting
        Waiting.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Waiting.tStart = globalClock.getTime(format='float')
        Waiting.status = STARTED
        thisExp.addData('Waiting.started', Waiting.tStart)
        Waiting.maxDuration = None
        # keep track of which components have finished
        WaitingComponents = Waiting.components
        for thisComponent in Waiting.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Waiting" ---
        Waiting.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 4.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *waiting_title* updates
            
            # if waiting_title is starting this frame...
            if waiting_title.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                waiting_title.frameNStart = frameN  # exact frame index
                waiting_title.tStart = t  # local t and not account for scr refresh
                waiting_title.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(waiting_title, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'waiting_title.started')
                # update status
                waiting_title.status = STARTED
                waiting_title.setAutoDraw(True)
            
            # if waiting_title is active this frame...
            if waiting_title.status == STARTED:
                # update params
                waiting_title.setText('Waiting...\n', log=False)
            
            # if waiting_title is stopping this frame...
            if waiting_title.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > waiting_title.tStartRefresh + 4-frameTolerance:
                    # keep track of stop time/frame for later
                    waiting_title.tStop = t  # not accounting for scr refresh
                    waiting_title.tStopRefresh = tThisFlipGlobal  # on global time
                    waiting_title.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'waiting_title.stopped')
                    # update status
                    waiting_title.status = FINISHED
                    waiting_title.setAutoDraw(False)
            # Run 'Each Frame' code from waiting_code
            total = trials.nTotal
            current = trials.thisN + 2
            
            if has_both_sides:
                total_display = total // 2
                current_display = (trials.thisN // 2) + 1
            else:
                total_display = total
                current_display = current
            
            waiting_title.text = f"Trial {current_display} / {total_display}"
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Waiting,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Waiting.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Waiting.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Waiting" ---
        for thisComponent in Waiting.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Waiting
        Waiting.tStop = globalClock.getTime(format='float')
        Waiting.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Waiting.stopped', Waiting.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Waiting.maxDurationReached:
            routineTimer.addTime(-Waiting.maxDuration)
        elif Waiting.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.000000)
        # mark thisTrial as finished
        if hasattr(thisTrial, 'status'):
            thisTrial.status = FINISHED
        # if awaiting a pause, pause now
        if trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials'
    trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Done" ---
    # create an object to store info about Routine Done
    Done = data.Routine(
        name='Done',
        components=[done_title],
    )
    Done.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Done
    Done.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Done.tStart = globalClock.getTime(format='float')
    Done.status = STARTED
    thisExp.addData('Done.started', Done.tStart)
    Done.maxDuration = None
    # keep track of which components have finished
    DoneComponents = Done.components
    for thisComponent in Done.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Done" ---
    Done.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *done_title* updates
        
        # if done_title is starting this frame...
        if done_title.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            done_title.frameNStart = frameN  # exact frame index
            done_title.tStart = t  # local t and not account for scr refresh
            done_title.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(done_title, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'done_title.started')
            # update status
            done_title.status = STARTED
            done_title.setAutoDraw(True)
        
        # if done_title is active this frame...
        if done_title.status == STARTED:
            # update params
            pass
        
        # if done_title is stopping this frame...
        if done_title.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > done_title.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                done_title.tStop = t  # not accounting for scr refresh
                done_title.tStopRefresh = tThisFlipGlobal  # on global time
                done_title.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'done_title.stopped')
                # update status
                done_title.status = FINISHED
                done_title.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Done,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Done.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Done.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Done" ---
    for thisComponent in Done.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Done
    Done.tStop = globalClock.getTime(format='float')
    Done.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Done.stopped', Done.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if Done.maxDurationReached:
        routineTimer.addTime(-Done.maxDuration)
    elif Done.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    # Run 'End Experiment' code from stimuli_code
    if stream is not None:
        try:
            stream.stop_stream()
        except Exception:
            pass
        stream.release_session()
    
    if X_list:
        from pathlib import Path
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(_thisDir).parents[1] / 'data' / 'raw'
        out_dir.mkdir(parents=True, exist_ok=True)
        ml = min(x.shape[1] for x in X_list)
        X = np.stack([x[:, :ml] for x in X_list])
        y = np.array(y_list, dtype=np.int64)
        np.save(out_dir / f"X_{ts}.npy", X)
        np.save(out_dir / f"y_{ts}.npy", y)
        print(f"Saved {X.shape[0]} trials — X={X.shape}")
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
