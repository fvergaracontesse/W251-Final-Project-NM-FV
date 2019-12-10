## Cloud configuration

### Create virtual machine in ibm cloud
```
ibmcloud sl vs create --datacenter=lon04 --hostname=v100a --domain=dima.com --image=2263543 --billing=hourly  --network 1000 --key=1558644 --flavor AC2_8X60X100 --san

```
### Install kaggle api
```
pip install kaggle --upgrade

```
### Create api key

```
kaggle.json
cp kaggle.json ~/.kaggle/.

```

### Download datasets

```
kaggle datasets download rtatman/handwritten-mathematical-expressions

```

### Copy dataset to input folder

```
mkdir input

cp -r CROHME_training_2011 input/.

```

### If using pipenv 

```
pipenv shell
pipenv install

```

### Option 1: Mount bucket to add models

```
#Install s3fs 
apt update && apt install -y s3fs

#Create directory to add models
mkdir  -m 777 /models

#Add credentials
echo "E2Pwm8aQxZcXJleMzPjn:FqJefRvlMtFmYJM3E2Cld5XPrg8Y3nrLjwdsyAxO" > /root/.cos_creds

#Retrict access to credential file
chmod 600 /root/.cos_creds

#Mount bucket to directory
s3fs final-project-w251-nm-fv /models -o passwd_file=/root/.cos_creds -o sigv2 -o use_path_request_style -o url=https://s3.private.us.cloud-object-storage.appdomain.cloud -o nonempty

#Copy models to bucket

cp -r models/* /models/

```
### Option 2: Use ibmcloud cli

```
ibmcloud cos config hmac

ibmcloud cos config auth

#Download

ibmcloud cos download --bucket hwe-w251-nm-fv --key KEY OUTFILE

#Upload

ibmcloud cos put-object --bucket hwe-w251-nm-fv --key KEY --body PATH/Filename

```
### Run pipeline

```
pipenv install
pipenv shell
python run_pipeline.py

```

## Jetson TX2 - Edge Configuration for predictions

1.Build docker image
```
docker build -t final_project_edge -f Dockerfile.edge .
```
2.Start container
```
docker run --rm --privileged -v /data:/data -p 8000:8000 -ti final_project_edge:latest bash
```
3. Clone repo within the container
```
git clone https://github.com/fvergaracontesse/W251-Final-Project-NM-FV.git
```
4. Install keras
```
pip3 install keras
```
5. Modify mnist.py to run with python instead of pipenv modifying shebang on the top to python.
6. Got to webapp folder on the repo and run
```
run python3 -m http.server --cgi 8000
```

## Ubuntu 18 - Configuration for predictions

1. Clone repository
```
git clone https://github.com/fvergaracontesse/W251-Final-Project-NM-FV.git

```
2. Go to main folder and run
 ```
 pipenv install
 pipenv shell
 pipenv run python3 -m http.server --cgi 8000
 ```






