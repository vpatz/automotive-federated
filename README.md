# Federated Learning for Automotive 

## Use case 1: EV battery range prediction model update using Federated Learning

### Model inputs

1. State of Charge (SOC) (%)
2. Battery Voltage (V)
3. Battery Temperature (C)
4. Current Vehicle Speed (km/h or mph)
5. Average Speed over the last 5 minutes
6. Current draw (A) (can be inferred or included directly)
7. State of Health (SOH) (%) (optional, useful for long-term accuracy)

### Model output

- Range (distance in km/miles)


### Simulated data for training

#### Input data generated as normalized (zero mean, unit variance) tensor 

    EVdata = torch.randn(NUM_SAMPLES, INPUT_DIM) # 100 samples, 10 features each

#### Output data is range in between (MIN_RANGE, MAX_RANGE)
    MIN_RANGE = 0
    MAX_RANGE = 100
    EVrange = torch.FloatTensor(NUM_SAMPLES,1 ).uniform_(MIN_RANGE, MAX_RANGE) 


### Model training (without Federated Learning)

    python3 model_trainer.py

It is expected that model weights will not converge during training as randomized data is used. 

### Model training using [`flower`](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html) FL library with `pytorch` model

Following steps are required to run the demo for EV range model training using Federated Learning 

#### Code changes

1. Use `flwr new` to create a template director 
2. Update model architecture and data source for the use case in `./ev_range_pred/task.py`
    - Replace the CIFAR dataset with random data 
    - Replace `flower_datasets` with `torch DataLoader`

Above steps are completed in `./ev_range_pred` directory.

#### Running the demo on `localhost`

Run below commands in different terminals as given in deployment [tutorial](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html).  


##### Terminal 1 - Run superlink

    flower-superlink --insecure

##### Terminal 2 - Run supernode

    flower-supernode \
         --insecure \
        --superlink 127.0.0.1:9092 \
        --clientappio-api-address 127.0.0.1:9094 \
        --node-config "partition-id=0 num-partitions=2"


##### Terminal 3 - Run client (to be ported to run on the vehicle edge ECU) 

    flower-superexec \
        --insecure \
        --plugin-type clientapp \
        --appio-api-address zupernode-1:9094

