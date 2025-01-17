

tau = 0.01
critic_target_weights = [1.0, 2.0, 3.0]
critic_weights = [1.0, 1.0, 1.0]

critic_target_weights = [x * tau + y * (1-tau) for x, y in zip(critic_target_weights, critic_weights)]

print(critic_target_weights)