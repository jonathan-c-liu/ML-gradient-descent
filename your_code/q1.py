from your_code import load_data
from your_code import GradientDescent
import numpy as np

trainF, testF, trainT, testT = load_data('mnist-binary')
learner = GradientDescent(loss='hinge', learning_rate=1e-4)
learner.fit(trainF, trainT, batch_size=1, max_iter=1000)