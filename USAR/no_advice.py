from runner import Runner
from common.arguments import get_common_args, get_coma_args
from env.hetro_usar_simple_4_room import Hetro_USAR_Simple_4_Room_Env
from env.hetro_usar_14_room import Hetro_USAR_14_Room_Env

import experiment_parameter as parameter


if __name__ == '__main__':
    for i in range(1):
        args = get_common_args()
        if args.alg.find('coma') > -1:
            args = get_coma_args(args)
        if parameter.env_type == 'Simple_4_Room':
            env = Hetro_USAR_Simple_4_Room_Env()
        elif parameter.env_type == '14_Room':
            env = Hetro_USAR_14_Room_Env()
        else:
            assert False==True, 'Unknown Environment'

        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]

        print ('env_info', env_info)
        runner = Runner(env, args)
        if not args.evaluate:
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
