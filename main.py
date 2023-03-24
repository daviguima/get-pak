import time
import argparse
from getpak.commons import Utils as u


def main():
    '''
    GET-pak entrypoint
    '''
    print('''
        _..._
      .'     '.      _
     /    .-""-\   _/ \ 
   .-|   /:.   |  | l | 
   |  \  |:.   /.-'-./ 
   | .-'-;:__.'    =/  ,ad8888ba,   88888888888  888888888888                                88
   .'=  *=|CNES _.='  d8"'    `"8b  88                88                                     88
  /   _.  |    ;     d8'            88                88                                     88
 ;-.-'|    \   |     88             88aaaaa           88            8b,dPPYba,   ,adPPYYba,  88   ,d8
/   | \    _\  _\    88      88888  88"""""           88  aaaaaaaa  88P'    "8a  ""     `Y8  88 ,a8"
\__/'._;.  ==' ==\   Y8,        88  88                88  """"""""  88       d8  ,adPPPPP88  8888[
         \    \   |   Y8a.    .a88  88                88            88b,   ,a8"  88,    ,88  88`"Yba,
         /    /   /    `"Y88888P"   88888888888       88            88`YbbdP"'   `"8bbdP"Y8  88   `Y8a
         /-._/-._/                                                  88
         \   `\  \                                                  88
          `-._/._/
                    ''')

    time.sleep(2) # Sleep for n seconds
    pass


if __name__ == '__main__':
    # ,--------------,
    # | Start timers |
    # '--------------'
    u.tic()
    t1 = time.perf_counter()
    # ,-----,
    # | RUN |
    # '-----'
    main()
    # ,------------------------------,
    # | End timers and report to log |
    # '------------------------------'
    t_hour, t_min, t_sec = u.tac()
    t2 = time.perf_counter()
    final_msg_1 = f'Finished in {round(t2 - t1, 2)} second(s).'
    final_msg_2 = f'Elapsed execution time: {t_hour}h : {t_min}m : {t_sec}s'
    print(final_msg_1)
    print(final_msg_2)
