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
pipenv shell
python run_pipeline.py

```

## Jetson TX2 - Edge Configuration




