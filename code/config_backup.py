# Based on: https://github.com/sunghoonhong/AirsimDRL/blob/master/config.py
reward = {
    'dsensor_close': -2.0,

    'miss': -0.1,
    # found reward
    'in': 3.0,
    'out': 1.0,
    #'small_in': 1.0,
    #'small_out': 0.5,

    # area reward, 10.0 is given if the drone successfully explore the whole area
    'area': 10.0,
    'none': 0,
    'near': -2.0,

    'success': 5.0,
    'dead': -5.0
}