# Federated Learning in Automotive

## Use case 1: EV battery range prediction model update 

### Model inputs (features extracted from a time window)



| Feature description  | Label | 
|---------|-------------|
|1. Initial State of Charge (SoC) (%)   | `soc_percent` |
|2. Delta State of Charge (DSoC) (%)    | `delta_soc_percent` |
|3. Total battery energy spend (kWh)    | `battery_energy_spend_kwh` |
|4. Regen energy gain (kWh)             | `regen_energy_gain_kwh` |
|5. Average speed  (km/h)               | `avg_speed_kmph` |
|6. Vehicle mass (kg)                 | `vehicle_mass_kg` |
|7. Distance travelled in time window (km) |   `distance_step_km` |
| Model output - Remaining vehicle range (km) |    `remaining_range_km` |


### Using data from EV Ramnge Simulator for training/validation

Using EV battery range simulator form [here.](https://github.com/vpatz/ev-distance-energy-sim) 
Features are typically avilable or can be calculated from raw data published in the vehicle CAN bus.



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

