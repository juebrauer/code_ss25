# 1. algorithm parameters
ALGO_MAP_COLOR_WALKABLE = (255,255,255)

# how much to go into the direction of the random new point?
# see code to activate:
# 1.) ALGO_INCR_DIST = absolute nr of pixels
# OR 
# 2.) ALGO_INCR_DIST = percentage to length to go on
#                      direction vector from nearest point
#                      towards this new point
ALGO_INCR_DIST = 40

ALGO_TERMINATION_RADIUS = 30

ALGO_WORLD_TO_USE = "world1.png"



# 2. visualization parameters: sizes, colors

# 2.1 sizes
VISU_LOCATION_START_RADIUS = 10
VISU_LOCATION_GOAL_RADIUS = 10
VISU_NODE_RADIUS = 5
VISU_LAST_RANDOM_POINT_RADIUS = 5
VISU_FOUND_PATH_WIDTH = 4

# 2.2 colors
VISU_COLOR_LOCATION_START = (255,0,0)
VISU_COLOR_LOCATION_GOAL = (0,0,255)
VISU_COLOR_TREE_NODE = (50,150,50)
VISU_COLOR_TREE_EDGE = (20,150,20)
VISU_COLOR_LAST_RANDOM_POINT = (0,0,0)
VISU_COLOR_TERMINATION_CIRCLE = (128,128,128)
VISU_COLOR_FOUND_PATH_EDGE = (255,0,0)
VISU_COLOR_FOUND_PATH_NODE = (200,0,0)

# 2.3 switches
VISU_SHOW_NODES = True



# 3. program

# run several steps at once if user presses
# a special key. but how many?
DEMO_RUN_N_STEPS = 500
