import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import os
import argparse

from keras_.models.housing_mlp import HousingMLP


# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


""" Parse the input Arguments """

script_cwd = os.path.dirname(__file__)
print("Script current working directory: ", script_cwd, flush=True)
parser = argparse.ArgumentParser()
parser.add_argument("--federation_rounds", default=3, type=int)
parser.add_argument("--learners_num", default=10, type=int)
parser.add_argument("--train_samples_per_learner", default=100, type=int)
parser.add_argument("--test_samples", default=100, type=int)
parser.add_argument("--nn_params_per_layer", default=1000, type=int)
parser.add_argument("--nn_hidden_layers_num", default=1, type=int)
parser.add_argument("--data_type", default="float32", type=str)

args = parser.parse_args()
print(args, flush=True)


""" Load training and test data """
required_training_samples = int(args.learners_num * args.train_samples_per_learner)
total_required_samples = required_training_samples + int(args.test_samples)

# /FlowerSource/flower/examples/experiments/keras/datasets/housing/original/data.csv
housing_np = pd.read_csv(script_cwd + "/keras_/datasets/housing/original/data.csv").to_numpy()
housing_np = housing_np[~np.isnan(housing_np).any(axis=1)]

total_rows = len(housing_np)
print(f"Debug: Total Rows Before: {total_rows}")

# Add dummy data at the end if the data.csv runs short of rows.
if total_required_samples > total_rows:
    diff = total_required_samples - total_rows
for i in range(diff):
    # Append random row from the original set of rows.
    housing_np = np.vstack(
        [housing_np,
            housing_np[random.randrange(total_rows)]])

print(f"Debug: Total Rows After: {total_rows}")

np.random.shuffle(housing_np)

train_data, test_data = housing_np[:required_training_samples], housing_np[:-args.test_samples]

print(f"training{len(train_data)} testing{len(test_data)}")

# First n-1 are input features, last feature is prediction/output feature.
x_train, y_train = train_data[:, :-1], train_data[:, -1:]
x_test, y_test = test_data[:, :-1], test_data[:, -1:]


# x_chunks -> List Of Size 10
# y_chunks -> List Of Size 10
x_chunks, y_chunks = np.split(x_train, args.learners_num), np.split(y_train, args.learners_num)

""" In the METIS code save the data splits on disk, so all the learners have access to data
(1) Create 10 files test_1.npz, test_2.npz, ..., test_x.npz ...
(2) 
"""



# Steps Performed by server. 
# 1. Select client for training. 
# 2. Send training instructions to client(s)
# 3. Client recieves those instructions and runs method.

class LearnerClient(fl.client.NumPyClient):

    def __init__(self, cid, model, x_train, y_train, x_test, y_test) -> None:
        super().__init__()
        self.cid = cid
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        import math
        # split_idx = math.floor(len(x_train) * 0.9)  # Use 10% of x_train for validation
        # self.x_test, self.y_test = x_train[split_idx:], y_train[split_idx:]
        self.x_test, self.y_test = x_test, y_test
      

    #Override
    def get_parameters(self, config):
        print(f"[Client {self.cid} get_parameters]")
        return self.model.get_weights()

    #Override
    def fit(self, parameters, config):
        print(f"\n[Client {self.cid}] fit, config: {config}")
        self.model.set_weights(parameters)

        # How do we split the x_train and y_train for the trainloaders?
        self.model.fit(self.x_train, self.y_train, epochs=4, batch_size=32, steps_per_epoch=3)
        return self.model.get_weights(), len(x_train), {}

    #Override
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        msc = self.model.evaluate(self.x_test, self.y_test)
        loss = 0.0
        return loss, len(self.x_test), {"msc" : float(msc)}


# Use the Virtual Client Engine to simulate N number of learners. 

def client_factory(cid: str) -> LearnerClient:
    """ Create a Flower Client representing a single organization """

    print(f"[Calling Client Factory for {cid}]")

    # Load Model
    model = HousingMLP(params_per_layer=args.nn_params_per_layer,
        hidden_layers_num=args.nn_hidden_layers_num,
        data_type=args.data_type).get_model()

    # Load the training data
    x_train = x_chunks[int(cid)] # Not too sure if this is correct!

    # Load the testing data
    y_train = y_chunks[int(cid)] # This needs to be the training data. 

    cid_x_test = x_test
    cid_y_test = y_test

    # Create a client object representing a single learner
    return LearnerClient(cid, model=model,x_train=x_train,y_train=y_train, x_test=cid_x_test, y_test=cid_y_test)


def main() -> None:
    # Start Flower simulation
    fl.simulation.start_simulation(
        client_fn=client_factory,
        num_clients=args.learners_num,
        client_resources={"num_cpus": 1},
        config=fl.server.ServerConfig(num_rounds=args.federation_rounds),
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=10,
            min_evaluate_clients=10,
            min_available_clients=args.learners_num,
        ),
    )


if __name__ == "__main__":
    main()