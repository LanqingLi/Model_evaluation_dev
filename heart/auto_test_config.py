# Auto test params
# Max distance between layers that can be considered as one plaque
LAYER_TOLERANCE = 3
# If the intersection index > VOL_THRESHOLD and other requirements are met,
# those two plaques will be considered as the same one
VOL_THRESHOLD = 0.3
# If distance of box center is greater than MAX_BOX_DISTANCE(unit: pixel), they won't be considered as the same plaque
MAX_BOX_DISTANCE = 30
# Judge if predicted plaque is the same as ground truth plaque
#       Mode 1: Categorical judgement, i.e. judge 100% ncP as one class and others as the other class. Two plaques
#               are of the same type if they belong to the same class.
#       Mode 2: Continuous judgement, i.e. Calculate calcium percentage of each plaque, with mP considered as 0.5 cP and
#               0.5 ncP. If difference in calcium percentage is greater than MAX_CAL_PERCENTAGE_DIFF, they will be
#               considered as different.
# Mode 2 is only available for plaque auto_test, not applicable for segment auto_test yet.
PLAQUE_COMPARISON_MODE = 1
MAX_CAL_PERCENTAGE_DIFF = 100
