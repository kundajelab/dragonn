---
layout: default
title: {{ site.name }}
---

# Getting Started 

## Launching an Amazon AWS instance with Dragonn 

1. Create an account with [Amazon AWS](<http://www.aws.amazon.com>)

..![AWS login window](https://github.com/annashcherbina/dragonn/blob/gh-pages-jekyll/images/aws_login.png "AWS Login Window")

2. Login to your AWS account, and set your region to "US West (Oregon)" or to "US West (Northern California)"

..![AWS Region](https://github.com/annashcherbina/dragonn/blob/gh-pages-jekyll/images/aws_region.png "AWS Select Region")

3. Go to the "services" tab and select "EC2" 

..![AWS Services](https://github.com/annashcherbina/dragonn/blob/gh-pages-jekyll/images/aws_services.png "AWS Services")

4. On the left pane of the Web site, select "AMIs", and type "dragonn" into the search bar. You should see AMI Name: "DragonnTutorial_KundajeLab" (if you are in the North California region) 
or "DragonnTutorial_KundajeLab_OREGON" if you are in the Oregon region. You will not see the AMI if your region is set to anything other than these two options (see step 2). 

..![AWS AMI](https://github.com/annashcherbina/dragonn/blob/gh-pages-jekyll/images/aws_ami.png "AWS AMI")

5. Select the AMI and click "Launch". 

6. Go through the setup tutorial  to configure your Dragonn instance, making sure to select the following options: 
..  a. In "Step 2: Choose an Instance Type" select "g2.2xlarge".  
....![AWS GPU Instance](https://github.com/annashcherbina/dragonn/blob/gh-pages-jekyll/images/aws_gpuinstance.png "AWS GPU Instance")
   
   

## Installing Dragonn Locally 
