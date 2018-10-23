import argparse
import json
import re
import smtplib
import subprocess

# Here are the email package modules we'll need
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def run_args(args, email_recipient=None):

    # Build command from args
    cmd = ['python', 'train.py']
    cmd += ['--{}={}'.format(k, v) for k, v in args.items()]

    # Spawn process to run the args
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if email_recipient:
        server = smtplib.SMTP('smtp.gmail.com', port=587)
        server.starttls()
        server.login('ct.job.runner', 'b00tcamp')

        msg = MIMEMultipart()
        msg['Subject'] = 'CT Job'
        # me == the sender's email address
        # family = the list of all recipients' email addresses
        msg['From'] = 'CT Job Runner'
        msg['To'] = email_recipient

        body = 'Return Code: {}\n'.format(completed.returncode)
        body += 'Stdout:\n' + completed.stdout.decode('utf-8') + '\n'
        body += 'Stderr:\n' + completed.stderr.decode('utf-8') + '\n'
        content = MIMEText(body, _subtype='text')
        msg.attach(content)

        server.sendmail('ct.job.runner@gmail.com', email_recipient, msg.as_string())
        server.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('args_path', type=str, help='Path to JSON file with args to run.')
    parser.add_argument('--email', type=str, default=None, help='Email address to notify when job completes.')
    script_args = parser.parse_args()

    # Validate
    if script_args.email is not None:
        email_re = re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')
        if not email_re.match(script_args.email):
            raise ValueError('Invalid email address: {}'.format(script_args.email))

    # Load args from args.json file
    with open(script_args.args_path, 'r') as json_file:
        args_ = json.load(json_file)

    print('Running job with args from {}...'.format(script_args.args_path))
    run_args(args_, email_recipient=script_args.email)
