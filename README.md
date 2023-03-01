# Credit_Card_Fraud_Detection_Wizengamot
Design Lab Project - IEM-CSE 2020

Team Name: Wizengamot

Project: Credit Card Fraud Detection (using machine learning)

Team Members: 
Ushnik Ghosh (Project Manager), 
Shuvozit Ghose (Lead Developer), 
Ushana Bandyopadhyay (Developer), 
Anindya Sundar Patwari (Developer), 
Susmit Sarkar (QA Manager), 
Aritra Khan (QA Tester)

Release 0.1

Master Branch:

Control Classes:
ProcessController.py - Central control class that controls all sub-processes, 
SMSController.py - Control class to send SMSs and read contents from SMSs

Entity Classes:
MLModel.py - To run the Machine Learning Model for Fraud Detection, 
Trainer.py - To train the Machine Leaning Model

Bugs:

Major Bugs(1) - Security message gets sent to SPAM folder in user's device.

Non-functional Requirements:

Email used instead of SMS due to unavailability of funds to buy SMS service.

Release 0.2

Master Branch:

Control Classes:
ProcessController.py - Central control class that controls all sub-processes (All control classes merged into one for efficiency)

Entity Classes:
Trainer.py - To train the Machine Leaning Model

Bugs:

Major Bugs(0) 

Bug Fixes(1) - issue: Security message gets sent to SPAM folder in user's device.  status: fixed

Release 1.0

Master Branch:

Control Classes:
ProcessController.py - Central control class that controls all sub-processes

Entity Classes:
Trainer.py - To train the Machine Leaning Model

Bugs:

Major Bugs(0) 

Bug Fixes(0)
