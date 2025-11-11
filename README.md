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

### Model training using [`flower`](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html) FL library with `pytorch` model

Following steps are required to run the demo for EV range model training using Federated Learning

- Use `flwr new` to create a template 