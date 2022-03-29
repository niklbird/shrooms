import pysftp
import paramiko
from scp import SCPClient
from os.path import expanduser
import os

'''
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
This script is not part of the product but only intended as a means of updating data on the server
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
'''

def createSSHClient(server, port, user, pubkey):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user,key_filename=pubkey)
    return client

home = expanduser("~")
user_name = os.getlogin().lower()
ssh = createSSHClient("188.68.49.103", 50001, user_name, home + "/.ssh/id_rsa")
scp = SCPClient(ssh.get_transport())
scp.put("./testus.txt", f"/home/{user_name}/testus.txt")
