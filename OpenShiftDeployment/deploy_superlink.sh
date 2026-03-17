# step-1 - deploy the superlink deployment
#oc apply -f superlink-deployment.yaml

# step-2 -  see deployments
oc get deployments


# step-3 - expose the superlink deployment to create service
#oc expose deployment flask-test-server --port=5000
oc expose deployment superlink --port=9095


# step-4 - create route to the service
oc create route passthrough --service=superlink --port=9095
