# Based on: https://github.com/sunghoonhong/AirsimDRL/blob/master/config.py
reward = {

    'miss': -0.1,
    # found reward
    'in': 10.0,
    'out': 5.0,
    #'small_in': 1.0,
    #'small_out': 0.5,

    # area reward, 100.0 is given if the drone successfully explore the whole area
    'area': 10.0,
    'none': 0,
    'near': -1.0,
    'dsensor_close': -1.0,
    'out_small': -5.0,

    'success': 20.0,
    'dead': -20.0
}