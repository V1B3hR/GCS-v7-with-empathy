import tensorflow as tf
import logging

class OnlineLearningModule:
    """
    The heart of the adaptive system. Applies incremental learning updates to the
    foundational model based on real-time user feedback.
    """
    def __init__(self, model: tf.keras.Model, config: dict):
        self.model = model
        self.config = config
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config['online_learning_rate'])
        logging.info(f"Online Learning Module Initialized with LR: {config['online_learning_rate']}")

    @tf.function
    def reinforcement_update(self, state, action_index, reward):
        """
        Performs a policy gradient update for implicit feedback (Reinforcement Learning).
        This gently nudges the model towards actions that don't get corrected.
        """
        with tf.GradientTape() as tape:
            mi_output, _, _ = self.model(state, training=False)
            prob_of_action = mi_output[0, action_index]
            loss = -tf.math.log(prob_of_action + 1e-9) * reward
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        logging.info(f"[LEARN] Reinforcement update applied. Reward: {reward}, Loss: {loss:.4f}")

    def corrective_update(self, state, correct_action_index):
        """
        Performs a single supervised learning step for explicit error correction.
        This is a more powerful update that directly teaches the correct label.
        """
        # We only want to update the primary MI output, not the adversary
        loss = self.model.train_on_batch(
            state,
            {"mi_output": tf.constant([[correct_action_index]]), "adversary_output": tf.zeros((1, self.config["train_subjects"]))},
            return_dict=True
        )
        logging.warning(f"[LEARN] Corrective update applied. MI Loss: {loss['mi_output_loss']:.4f}")
