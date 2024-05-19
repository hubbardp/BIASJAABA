import pygame
import numpy as np
import urllib.request, json

# how much to scale the video frame by
scale = 1.0
# how much of a border to add to the screen after the edge of the arena
# not scaled
border = 10.0

# set the projection size - matches frame size
screenSz = (scale*960,scale*960)

# how we will transform tracking results to projection
transform = lambda x: scale*x

# plotting parameters
bkgdColor = pygame.Color(255,255,255,255)
dotColor = pygame.Color(0,0,0,255)
flyColor = pygame.Color(123,123,123,255)
arenaColor = pygame.Color(0, 0, 0, 255)
arenaLineWidth = 2

# flags to make pygame faster
pygameFlags = pygame.DOUBLEBUF
bpp = 8

# size of a dot -- currently we are doing a circle, and the radius is the semimajor axis length
dotA = 11.77
dotB =  4.01

# for closed-loop experiments: urls to communicate with BIAS
getBIASUrl = lambda s: 'http://127.0.0.1:5010/?plugin-cmd={%22plugin%22:%22FlyTrack%22,%22cmd%22:%22'+s+'%22}'
flyTrackUrl = getBIASUrl('get-last-clear-track')
arenaParamsUrl = getBIASUrl('get-arena-params')


def OpenLoopCircle():
    """
    OpenLoopCircle()
    Draw a dot that goes around in a circle
    """

    # How far from arena center is the dot's trajectory
    dotDistCenter = 300

    # how fast to update the screen
    screenFPS = 60.
    
    # how fast should the dot go
    dtheta_dt_deg_per_sec = 30
    dtheta_dt = dtheta_dt_deg_per_sec * np.pi / 180.0

    # hard-coded arena location
    arena = {'x': 466.0, 'y': 483.0, 'r': 433.0}
    arenaPos = (transform(arena['x']),transform(arena['y']))
    arenaRadius = arena['r']*scale
    
    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode(screenSz)
    clock = pygame.time.Clock()
    running = True

    # inital time
    t = 0.0
    theta = 0.0

    # size of circle to plot
    dotRadius = dotA*scale
    # initialize dot center
    dotPos = pygame.Vector2(transform(arena['x'] + dotDistCenter*np.cos(theta)),
                            transform(arena['y'] + dotDistCenter*np.sin(theta)))

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # fill the arena with a color to wipe away anything from last frame
        screen.fill(bkgdColor)

        # draw the arena circle
        pygame.draw.circle(screen, arenaColor, arenaPos, arenaRadius,
                           width=arenaLineWidth)

        # draw the stimulus
        pygame.draw.circle(screen, dotColor, dotPos, dotRadius)

        # update position
        theta = dtheta_dt * t
        dotPos.x = transform(arena['x']+dotDistCenter*np.cos(theta))
        dotPos.y = transform(arena['y']+dotDistCenter*np.sin(theta))    

        # flip() the display to put your work on screen
        pygame.display.flip()

        # limits FPS to minDt
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(screenFPS) / 1000
        t+=dt

    pygame.quit()

def ClosedLoop(stimFun=None,debugPlotFly=False):
    """
    ClosedLoopOnFly(stimFun=None,debugPlotFly=False)
    Get the current position of the fly from BIAS. 
    Plot the stimulus dot in a position relative to it, 
    as defined by input function stimFun. 
    stimFun inputs the ellipse pose of the tracked fly 
    as a dict. It modifies these parameters in place to 
    set the desired location of the stimulus. 
    """
    
    # get arena location
    with urllib.request.urlopen(arenaParamsUrl) as url:
        data = json.load(url)
    if len(data) == 0:
        print('Could not get arena parameters')
        return
    data = data[0]
    if not data['success']:
        print('Could not get arena parameters')
        return
    data = json.loads(data['value'])
    arena = {'x': data['x'], 'y': data['y'], 'r': data['a']}
    arenaPos = (transform(arena['x']),transform(arena['y']))
    arenaRadius = arena['r']*scale

    # size of the stimulus
    dotRadius = dotA*scale
    dotPos = pygame.Vector2(0,0)

    # plot the fly for debugging
    if debugPlotFly:
        flyRadius = dotA/2*scale
        flyPos = pygame.Vector2(0,0)
    
    # keep track of how many frames we skip
    maxFrameSkipRecord = 10
    freqSkipFrame = np.zeros(maxFrameSkipRecord)
    currFrame = np.nan

    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode(screenSz,pygameFlags,bpp)
    clock = pygame.time.Clock()
    running = True

                
    # fill the arena with a color to wipe away anything from last frame
    screen.fill(bkgdColor)

    # draw the arena
    pygame.draw.circle(screen, arenaColor, arenaPos, arenaRadius,
                       width=arenaLineWidth)

    isFirst = True
    
    while running:

        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # erase the dot
        if not isFirst:
            pygame.draw.circle(screen, bkgdColor, dotPos, dotRadius*2)
            if debugPlotFly:
                pygame.draw.circle(screen, bkgdColor, flyPos, flyRadius*2)                
        isFirst = False

        # get current fly position
        with urllib.request.urlopen(flyTrackUrl) as url:
            data = json.load(url)
        if len(data) == 0:
            continue
        data = data[0]
        if not data['success']:
            continue
        data = json.loads(data['value'])

        # keep track of difference in frames
        lastFrame = currFrame
        currFrame = data['frame']
        dFrame = currFrame - lastFrame

        if debugPlotFly:
            flyPos.x = transform(data['x'])
            flyPos.y = transform(data['y'])
            pygame.draw.circle(screen, flyColor, flyPos, flyRadius)
        
        # apply optional transformation to modify the stimulus location
        if stimFun is not None:
            stimFun(data)

        # move the stimulus dot
        dotPos.x = transform(data['x'])
        dotPos.y = transform(data['y'])

        # store number of frames skipped
        if dFrame > maxFrameSkipRecord:
            dFrame = maxFrameSkipRecord
        if not np.isnan(dFrame) and dFrame >= 1:
            freqSkipFrame[dFrame-1] += 1
                
        # fill the arena with a color to wipe away anything from last frame
        #screen.fill(bkgdColor)

        # draw the arena
        #pygame.draw.circle(screen, arenaColor, arenaPos, arenaRadius,
        #                   width=arenaLineWidth)

        # draw the dot dot
        pygame.draw.circle(screen, dotColor, dotPos, dotRadius)

        # flip() the display to put your work on screen
        pygame.display.flip()

        # limits FPS to minDt
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        #dt = clock.tick(minDt) / 1000
        
    pygame.quit()

    # print stats on how many frames are skipped
    print("Frame skip histogram:")
    z = np.sum(freqSkipFrame)
    for i in range(maxFrameSkipRecord):
        print(f'{i+1}: {freqSkipFrame[i]/z}')

def inFront(data):
    """
    inFront(data)
    Set stimulus position to be 10 pixels in front of the tracked fly.
    """
    offset = 50
    dx = offset*np.cos(data['theta'])
    dy = offset*np.sin(data['theta'])
    data['x'] += dx
    data['y'] += dy
    
if __name__ == "__main__":
    ClosedLoop(inFront,True)
    #OpenLoopCircle()

