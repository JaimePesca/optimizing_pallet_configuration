def parse_args():
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('method', metavar='method', 
                        type=str, choices=['rl', 'bl', 'baf', 'bssf', 'blsf'], 
                        help='choose the method from {"rl", "bl", "baf", "bssf", "blsf"}.')
    
    parser.add_argument('lookahead', metavar='lookahead', 
                        type=int,
                        help='choose the lookahead value.')
    
    parser.add_argument('--data', metavar='', 
                        type=str, default='generated', choices=['generated', 'input', 'file'], 
                        help='choose the input source from {"generated", "input", "file"} (default: generated).')
    
    parser.add_argument('--path', metavar='', 
                        type=str, default=None, 
                        help='set the file path, only used if --data is "file" (default: None).')
    
    parser.add_argument('--n_iterations', metavar='', 
                        type=int, default=100, 
                        help='set the number of iterations, only used if --data is "generated" (default: 100).')
    
    parser.add_argument('--seed', metavar='', 
                        type=str, default=None, 
                        help='set the random seed for reproducibility, only used if --data is "generated" (default: None).')
    
    parser.add_argument('--verbose', metavar='', 
                        type=int, default=1, 
                        help='set verbose level (default: 1).')
    
    parser.add_argument('--train', 
                        action='store_true', 
                        help='enable training mode, only used if method is "rl" (default: False).')
    
    parser.add_argument('--batch_size', metavar='', 
                        type=int, default=32, 
                        help='set batch_size, only used if train is True (default: 32).')
    
    parser.add_argument('--visualize', 
                        action='store_true', 
                        help='enable visualization mode (default: False).')
    
    return parser.parse_args()

import numpy as np
import os, shutil, time

from env import *
from agent import *

import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

heuristics = {
    'bl': bottom_left,
    'baf': best_area_fit, 
    'bssf': best_short_side_fit, 
    'blsf': best_long_side_fit, 
}

def deeppack3d(method, lookahead, *, n_iterations=100, seed=None, verbose=1, data='generated', path=None, train=False, visualize=False, batch_size=32):
    reset_rng(seed)
    
    # ==================================================
    # DEFINIMOS LOS 65 BINS DEL 747-400F (ESCALA ENTERA)
    # ==================================================
    bin_sizes = []
    bin_sizes += [(24, 24, 32)] * 30   # 30 pallets M1
    bin_sizes += [(6, 12, 32)] * 23   # 23 slices superiores M1H
    bin_sizes += [(24, 16, 32)] * 9    # 9 pallets P6P
    bin_sizes += [(15, 16, 23)] * 2    # 2 contenedores LD1
    bin_sizes += [(20, 20, 18)] * 1    # 1 bulk

    # Environment:
    # - n_bins = 1  -> solo 1 bin activo a la vez
    # - max_bins = len(bin_sizes) -> hasta 65 bins secuenciales
    # - bin_sizes = lista de tamaños reales
        # Environment:
    # Usamos 65 bins simultáneos, uno por cada posición física del avión
    global_size = (55, 24, 32)   # tamaño de referencia para RL (el más grande aprox)

    env = MultiBinPackerEnv(
        n_bins=len(bin_sizes),        # ⬅️ 65 bins simultáneos
        size=global_size,
        max_bins=len(bin_sizes),      # ⬅️ no queremos más de 65
        k=lookahead,
        prealloc_items=100,
        verbose=verbose,
        bin_sizes=bin_sizes           # ⬅️ cada packer tiene su propio (W,H,D)
    )

    if data == 'file':
        env.conveyor = FileConveyor(k=env.k, path=path).reset()
    elif data == 'input':
        env.conveyor = InputConveyor(k=env.k).reset()

    if visualize:
        if os.path.exists('./outputs'):
            shutil.rmtree('./outputs')
        os.makedirs('./outputs')

    if train:
        print(f'Training with method "{method}" and lookahead {lookahead}...')
        
        if method != 'rl':
            raise Exception('training mode can only be used if method is "rl"')

        # env = BinPackerEnv(size=(32, 32, 32), k=env.k, bin_size=(32, 32, 32))
        agent = Agent(env, train=True, verbose=verbose > 0, visualize=visualize, batch_size=batch_size)

        agent.eps = 1.0
        for i in range(n_iterations):
            print(f'Iteration {i}')
            start_time = time.time()
            yield from agent.run(100, verbose=verbose > 1)
            agent.eps = max(agent.eps * 0.95, 0.025)
            
        data_arr = np.asarray([utils for utils, n_bins, ep_reward in agent.ep_history])
        # y = np.ones(100)
        # data = np.convolve(data, y, 'valid') / len(y)
        sns.lineplot(data=data_arr)
        plt.savefig(f'./util.jpg')
        plt.show()
        
        data_arr = np.asarray([ep_reward for utils, n_bins, ep_reward in agent.ep_history])
        # y = np.ones(100)
        # data = np.convolve(data, y, 'valid') / len(y)
        sns.lineplot(data=data_arr)
        plt.savefig(f'./ep_reward.jpg')
        plt.show()

        import uuid
        uid = uuid.uuid4()
        print(f'saved model at ./{uid}.h5')
        agent.q_net.save(f'{uid}.h5')
    else:
        if verbose > 0:
            print(f'Testing with method "{method}" and lookahead {lookahead}.')
        
        if method == 'rl':
            model_path = f'./models/k={lookahead}.h5'
            agent = Agent(env, train=False, verbose=verbose > 0, visualize=visualize, batch_size=batch_size)
            agent.q_net = tf.keras.models.load_model(model_path, compile=False)
            agent.eps = 0.0
        else:
            agent = HeuristicAgent(heuristics[method], env, verbose=verbose > 0, visualize=visualize)
        
        start_time = time.time()
        
        try:
            # colocaciones (placements)
            yield from agent.run(n_iterations, verbose=verbose > 1)
        except Exception as e:
            if np.all(np.array(env.conveyor.reset().peek()) == None):
                if verbose > 0:
                    print('\n=====the end of conveyor line=====')
            else:
                print(e)

        # 👉 NUEVO: al final devolvemos el env para poder leer used_packers
        yield env

        if verbose > 0:
            print()
            next_items = np.array(env.conveyor.reset().peek()).tolist()
            avg_util = np.mean([util for utils, n_bins, ep_reward in agent.ep_history[:] for util in utils[:]])
            used_items = np.sum([n_bins for utils, n_bins, ep_reward in agent.ep_history[:] for util in utils[:]])
            
            print(f'Used time: {int(time.time() - start_time)} seconds')
            print(f'Next items: {next_items}')
            print(f'Average space util: {avg_util}')
            print(f'Used bins: {used_items}')


def main():
    args = parse_args()
    
    reset_rng(args.seed)

    for _ in deeppack3d(args.method, 
                        args.lookahead, 
                        n_iterations=args.n_iterations, 
                        seed=args.seed, 
                        train=args.train, 
                        verbose=args.verbose, 
                        data=args.data, 
                        path=args.path,
                        visualize=args.visualize, 
                        batch_size=args.batch_size):
        pass

if __name__ == "__main__":
    main()
