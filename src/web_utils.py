import paramiko
from scp import SCPClient
from os.path import expanduser
import os
import constants

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


def send_file_to_server(filename_l, filename_s):
    home = expanduser("~")
    user_name = os.getlogin().lower()
    ssh = createSSHClient("188.68.49.103", 50001, user_name, home + "/.ssh/id_rsa")
    scp = SCPClient(ssh.get_transport())
    scp.put(filename_l, f"/home/{user_name}/shroom_data/{filename_s}")


def default_file_send():
    send_file_to_server(constants.pwd + f'/web/update_data.txt', "update_data.txt")
    send_file_to_server(constants.pwd + f'/web/update_data_grainy.txt', "update_data_grainy.txt")
    send_file_to_server(constants.pwd + f'/web/update_file.json', "update_file.json")
