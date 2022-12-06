import json
import boto3
import os
import email
from email.parser import BytesParser
from email import policy
import numpy as np
import sms_spam_classifier_utilities as utilities

s3 = boto3.client("s3")
bucket = "lixuanyanghw3-emails"


def lambda_handler(event, context):

    # extract msg content
    file_obj = event["Records"][0]
    filename = str(file_obj["s3"]['object']['key'])
    fileObj = s3.get_object(Bucket = bucket, Key=filename)
    data = fileObj['Body'].read()
    #print("data:",data)
    msg = BytesParser(policy=policy.SMTP).parsebytes(data)
    #print("message:", msg)
    plain = msg.get_body(preferencelist=('plain'))
    plain = ''.join(plain.get_content().splitlines(keepends=True))
    plain = '' if plain == None else plain
    #print("plain:",plain)
    msgbody = plain.replace('\r\n',' ').strip()
    print("msg body strip:",msgbody)
    
    #predict
    runtime = boto3.client('runtime.sagemaker')
    ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
    print('endpoint,',ENDPOINT_NAME)
    vocabulary_length = 9013
    test_messages = [msgbody]
    #test_messages = ["FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop"]
    one_hot_test_messages = utilities.one_hot_encode(test_messages, vocabulary_length)
    encoded_test_messages = utilities.vectorize_sequences(one_hot_test_messages, vocabulary_length)
    msgobj = json.dumps(encoded_test_messages.tolist())
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,ContentType='application/json',Body=msgobj)
    response_body = response['Body'].read().decode('utf-8')
    result = json.loads(response_body)
    print(result)
    
    pred = int(result['predicted_label'][0][0])
    if pred == 1:
        CLASSIFICATION = "SPAM"
    elif pred == 0:
        CLASSIFICATION = "HAM"
    
    CLASSIFICATION_CONFIDENCE_SCORE = str(float(result['predicted_probability'][0][0]) * 100)
    
    #send response email
    
    #print('classification:',CLASSIFICATION)
    #print('score:',CLASSIFICATION_CONFIDENCE_SCORE)
    email_body = msgbody[:240]
    #print("msg body:",msgbody[:240])
    sender = msg['From'].split('<')[1].split('>')[0]
    #print("sender:", msg['From'].split('<')[1].split('>')[0])
    date = msg['Date']
    #print("msg receive date:",msg['Date'])
    email_subjet = msg['Subject']
    #print("msg subject:",msg['Subject'])
    response_SUBJECT = "SPAM CHECK RESPONSE"
    response_BODY_TEXT = "We received your email sent at " + date + " with the subject " + email_subjet + ".\r\nHere is a 240 character sample of the email body:\r\n" + email_body + "\r\nThe email was categorized as " + CLASSIFICATION + " with a " + CLASSIFICATION_CONFIDENCE_SCORE + "% confidence."
    CHARSET = "UTF-8"
    client = boto3.client('ses',region_name="us-east-1")
    print(str(msg['To']))
    #Provide the contents of the email.
    response = client.send_email(
        Destination={
            'ToAddresses': [
                sender,
            ],
        },
        Message={
            'Body': {

                'Text': {
                    'Charset': CHARSET,
                    'Data': response_BODY_TEXT,
                },
            },
            'Subject': {
                'Charset': CHARSET,
                'Data': response_SUBJECT,
            },
        },
        Source=str(msg['To']),
        
    )


    
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
