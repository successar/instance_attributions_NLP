import paramiko
import subprocess

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("script_command", type=str)

args, unknown = parser.parse_known_args()

print("Running Command: ", args)

envvars = {}
for key, val in map(lambda x : x.split("="), unknown):
    envvars[key] = val

envvars_command = " ".join([f"{key}={val}" for key, val in envvars.items()])

print("Running with Env Vars")
print(envvars)
print(envvars_command)

subprocess.run(["git", "status"], check=True)

print("=" * 20)
if input("Do you want to Proceed ?").upper() != "Y":
    exit(1)

print("===== Pushing Local git to master =======")
subprocess.run(["git", "push", "origin", "master"], check=True)

try:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect("login.discovery.neu.edu", username="jain.sar")

    print("======== Running on Cluster =========")
    stdin, stdout, stderr = client.exec_command(
        f"cd influence_info_repo; git checkout master; git pull origin master; {envvars_command} {args.script_command}"
    )
    for line in stdout:
        print("... " + line.strip("\n"))

    import time

    time.sleep(5)

    client.close()
except:
    print("Error")
    client.close()

