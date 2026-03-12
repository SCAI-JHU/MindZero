import numpy as np

from agent import HumanAgent
from env import STAY, ConstructionEnv


def test_complete_task(num_tests):
    """
    测试智能体能够完整完成任务
    """
    print("=" * 70)
    print(f"测试完整任务执行（{num_tests}个随机任务）")
    print("=" * 70)

    rng = np.random.RandomState(2024)

    passed = 0
    failed = 0
    total_steps = []

    for test_id in range(num_tests):
        seed = rng.randint(0, 100000)

        try:
            env = ConstructionEnv(seed=seed)

            # 随机选择目标（从10个方块中选2个）
            goal_blocks = tuple(sorted(rng.choice(10, size=2, replace=False)))
            env.human_goal = goal_blocks

            agent = HumanAgent(env, goal=goal_blocks, seed=seed, priority=True)

            for step in range(200):
                obs = env._get_obs()
                action_human = agent.sample_action(obs)
                action_helper = STAY
                obs, done, info = env.step((action_human, action_helper))

                if done:
                    if info["goal_achieved"]:
                        passed += 1
                        total_steps.append(env.time_step)
                    else:
                        failed += 1
                        print(
                            f"  ✗ 测试 {test_id} 超时 (种子={seed}, 目标={goal_blocks})"
                        )
                    break

            if not done:
                failed += 1
                print(f"  ✗ 测试 {test_id} 未完成 (种子={seed}, 目标={goal_blocks})")

            # 每100个测试打印一次进度
            if (test_id + 1) % 100 == 0:
                avg_steps = sum(total_steps) / len(total_steps) if total_steps else 0
                print(f"  进度: {test_id + 1}/{num_tests}, 平均步数: {avg_steps:.1f}")

        except Exception as e:
            failed += 1
            print(f"  ✗ 测试 {test_id} 异常: {e}")

    avg_steps = sum(total_steps) / len(total_steps) if total_steps else 0

    print(f"\n{'='*70}")
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print(f"平均完成步数: {avg_steps:.1f}")
    if failed == 0:
        print("所有任务测试通过 ✓")
    print(f"{'='*70}")

    assert failed == 0, f"有 {failed} 个任务失败"


if __name__ == "__main__":
    test_complete_task(num_tests=1000)
