#!/usr/bin/env python3

from collections import defaultdict
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def f_index(f):
    return int(f.stem.rsplit('-', 1)[-1])


def main():
    if True:
        files = sorted(Path('/tmp/').glob('a1-*.txt'),
                       key=f_index)

        rewss = defaultdict(list)
        for f in files:
            eps_index = f_index(f)
            rewss['eps'].append(eps_index)
            with open(f, 'r') as fp:
                ls = fp.readlines()
                index = 0
                for l in ls:
                    if l.startswith('Average rewards per second:'):
                        break
                    index += 1
                print(index)

            rews = {}
            for l in ls[index + 1:index + 11]:
                a, b = l[3:].split(':', 1)
                rews[a.strip()] = float(b.strip())
            for k, v in rews.items():
                rewss[k].append(v)
    rewss_default = rewss

    if True:
        files = list(Path('/tmp/docker/').glob('rew_*.csv'))
        dfs = {f.stem: pd.read_csv(f).Value.to_numpy() for f in files}
        rewss = {}
        rewss['eps'] = pd.read_csv(next(iter(files))).Step.to_numpy()
        rewss.update(dfs)
        rewss = {k: v[::33] for k, v in rewss.items()}
        # df = pd.merge(dfs, how='left')
        # df = pd.concat(dfs, join='inner')
        # print(df)
    rewss_flat = rewss
    scales = dict(
        termination=-0.0,
        tracking_lin_vel=1.0,
        tracking_ang_vel=0.5,
        lin_vel_z=-2.0,
        ang_vel_xy=-0.05,
        orientation=-0.,
        # torques = -0.00001,
        dof_vel=-0.,
        dof_acc=-2.5e-7,
        base_height=-0.,
        feet_air_time=1.0,
        collision=-1.,
        feet_stumble=-0.0,
        action_rate=-0.01,
        stand_still=-0.,
        torques=-0.0002,
        dof_pos_limits=-10.0,
    )

    p = [k for k in rewss.keys() if k != 'eps']
    p = np.reshape(p, (2, 5))
    fig, ax = plt.subplot_mosaic(p)

    rew_totals = {}
    stp_totals = {}
    for name, rewss in {'plane': rewss_flat, 'default': rewss_default}.items():
        rew_totals[name] = sum([
            np.asarray(rewss[F'rew_{k}']) *
            scales[k] for k in scales if F'rew_{k}' in rewss])
        stp_totals[name] = rewss['eps']
        for k, v in rewss.items():
            if k == 'eps':
                continue
            # ax[k].plot(rewss['eps'], v, label=k)
            ax[k].plot(rewss['eps'], v, label=name)
            ax[k].set_title(k)
            ax[k].legend()
            ax[k].grid('on')
            # ax[k].set_xticklabels(rewss['eps'], rotation=45)
    fig.suptitle('Comparison')
    fig.supxlabel('epoch')
    fig.supylabel('reward')
    plt.tight_layout(pad=1.01)
    plt.subplots_adjust(
        left=0.125,
        bottom=0.1,
        right=0.9,
        top=0.9,
        wspace=0.3,
        hspace=0.2)
    plt.show()
    print(rew_totals['plane'].shape)
    print(rew_totals['default'].shape)

    plt.plot(stp_totals['plane'], rew_totals['plane'], label='plane')
    plt.plot(stp_totals['default'], rew_totals['default'], label='default')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('reward')
    plt.title('net reward comparison')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
