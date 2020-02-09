# Cluster Start

minikube start --cpus=3 --memory=3000mb

# chache images folder

cd ~minikube/cache ...

# kubectl

kubectl get nodes
NAME STATUS ROLES AGE VERSION
minikube Ready master 3m59s v1.17.2

kubectl get pods

# minicube comands

minikube stop
minikube start (-p name multiples cluster probar diferentes clusters versiones etc...)
minikube addons list (interesante el dashboard)
minikube dashboard -- lanza dashboard del cluster
minikube logs
minikube ip
minikube status

# folder ./kube/config

~/kube/config --> kubectl config del cluster donde se conecta

# pods

Es como la unidad minima a diferencia de un contendor de docker un pod envuelve el contendor es decir puede tener varios contenedores dentro de un pod (los contendores internos pueden comunicarse entre ellos)
Aunque es habitual un pod un container(para simplificar)

kubectl apply -f pod.yaml

kubectl get all

kubectl get pod flaskserver

kubectl describe pod flaskserver

## conecting to image

kubectl exec -it flaskserver bash

## removing one pod

kubectl delete pod flaskserver

kubectl delete -f pod.yaml

## conection to cluster

minikube ssh --> ahora tenemos visibilidad con los cpods/containers
-curl IP

## si queremos acceder desde fuera podemos hacer un port forwariding

kubectl port-forward apiflask 8080:80
