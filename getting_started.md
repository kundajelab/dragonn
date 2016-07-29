---
layout: default
title: {{ site.name }}
---

# Getting Started 

## Launching an Amazon AWS instance with Dragonn 

1. Create an account with [Amazon AWS](<http://www.aws.amazon.com>)
![AWS login window](https://github.com/annashcherbina/dragonn/blob/gh-pages-jekyll/images/aws_login.png "AWS Login Window")

2. Login to your AWS account, and set your region to "US West (Oregon)" or to "US West (Northern California)"

![AWS Region](https://github.com/annashcherbina/dragonn/blob/gh-pages-jekyll/images/aws_region.png "AWS Select Region")

3. Go to the "services" tab and select "EC2" 

![AWS Services](https://github.com/annashcherbina/dragonn/blob/gh-pages-jekyll/images/aws_services.png "AWS Services")

4. On the left pane of the Web site, select "AMIs", and type "dragonn" into the search bar. You should see AMI Name: "DragonnTutorial_KundajeLab" (if you are in the North California region) 
or "DragonnTutorial_KundajeLab_OREGON" if you are in the Oregon region. You will not see the AMI if your region is set to anything other than these two options (see step 2). 

![AWS AMI](https://github.com/annashcherbina/dragonn/blob/gh-pages-jekyll/images/aws_ami.png "AWS AMI")

5. Select the AMI and click "Launch". 

6. Go through the setup tutorial  to configure your Dragonn instance, making sure to select the following options: 
  a. In "Step 2: Choose an Instance Type" select "g2.2xlarge".  
  ![AWS GPU Instance](https://github.com/annashcherbina/dragonn/blob/gh-pages-jekyll/images/aws_gpuinstance.png "AWS GPU Instance")
  b. In "Step 3: Configure Instance Details" use the default options. 
  ![Step 3: Configure Instance Details](https://github.com/annashcherbina/dragonn/blob/gh-pages-jekyll/images/aws_step3.png "Step 3: Configure Instance Details")
  c. In "Step 4: Add Storage" set the Size(GiB) value to 20. 
  ![Step 4: Add Storage](https://github.com/annashcherbina/dragonn/blob/gh-pages-jekyll/images/aws_step4.png "Step 4: Add Storage")
  d. In "Step 5: Tag Instance" leave the defaults (expert users may want to add tags to identify the instance uniquely). 
  e. In "Step 6: Configure Security Group" follow these steps: 
     	i. Add Rule --> "Custom TCP Rule" ; Port Range --> 8443; Source --> Anywhere 
	ii. Add Rule --> HTTP; Port Range --> 80; Source --> Anywhere 
	iii. Add Rule --> HTTPS; Port Range --> 8443; Source --> Anywhere 
  f. In "Step 7: Review and Launch" leave all defaults and click "Launch". 
  g. In "Select an existing key pair or create a new key pair" select "Create a new key pair". Select a name for your key pair (such as "dragonn_keys") and click "Download Key Pair". 
     **Keep the file in a safe place, you will need it to connect to your Amazon AWS instance.** 
     
7. When you launch the instance, you will arrive back at the EC2 dashboard, and you will see your new instance in the "running" state. Please note that the cost of running your g2.2xlarge instance 
is $0.65 per hour. 

8. Select your instance, and click the "Connect" button. You will see instructions on how to connect to your instance through ssh (If you are running Windows, you will need an SSH client such as [putty]
(<http://www.chiark.greenend.org.uk/~sgtatham/putty/>). Follow the instructions in the popup window to chmod your key file (see step 6g) and ssh into your instance. It is likely that you will need to ssh in as the "ubuntu" user rather than the root user. Modify the example command to indicate "ssh -i 'dragonn_keys.pm' ubuntu@ec2-52-43-29-19.us-west-2.compute.amazonaws.com" 

9. Once you have connected to the instance, type 'ls' in the ubuntu home directory to learn the directory contents. You should see the following set of files (green) and directories (blue): 

10. refer to the file "README.txt" for instructions on how to run Dragonn from the command line. 

11. Alternatively, you can run the Dragonn jupyter notebook by executing the following commands: 
```
sudo su 
passwd ubuntu 
```
enter your desired password when prompted.

```
./launch_notebook.sh 
```
12. After you have launched the juypter server, in your browser, navigate to the public ip address of your GPU instance on port 80. You can find the public ip address from the EC2 dashboard: 


13. The username is "ubuntu", and the password is the same one that you created in step 11. 

14. Once you have logged in, you will see files associated with the dragonn softwre. Click on the "examples" folder. 

15. Click on "workshop_tutorial.ipynb" to launch the Jupyter notebook for the tutorial. 

When you are finished with the Amazon instance, follow these steps to shutdown the instance (and avoid incurring extra usage fees): 

1. Navigate to the EC2 dashboard and select your Amazon instance. 

2. In the toolbar, select "Actions"--> "Instance State" --> "Stop" 

3. Wait for your Amazon instance to shutdown. If you want to permanently delete the Amazon instance, select "Terminate" (instead of "Stop"). 

## Installing Dragonn Locally 
