import paramiko
import subprocess

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("script_command", type=str)
parser.add_argument("--sbatch", action="store_true")
parser.add_argument("--hours", type=str, default="01")
parser.add_argument("--y", action="store_true")
args, unknown = parser.parse_known_args()

envvars = {}
for key, val in map(lambda x: x.split("="), unknown):
    envvars[key] = val

envvars_command = " ".join([f"{key}={val}" for key, val in envvars.items()])

print("===== Running Command: ", args)

print("===== Running with Env Vars =====")
print(envvars)
print(envvars_command)

print("===== Will run on Discovery ======\n")

if "sbatch" not in args.script_command and args.sbatch :
    args.script_command = f"sbatch --time {args.hours}:00:00 discovery_files/gpu_sbatch.sh " + args.script_command

commands = [
    "cd influence_info_repo",
    "conda activate influence",
    "git checkout master",
    "git pull",
    f"{envvars_command} {args.script_command}"
]

command = "; ".join(commands)

print("\n".join(command.split("; ")))

print()

output = subprocess.run(["git", "status"], check=True, capture_output=True)
print(str(output.stdout.decode('utf-8')))

if 'influence_info' in str(output.stdout.decode('utf-8')) :
    print("Code change in repo; commit")
    msg = input("Enter msg if you want to commit : ")
    if len(msg.strip()) == 0 :
        exit(1)
    else :
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", msg.strip()], check=True)

print("=" * 20)
if (not args.y) and (input("Do you want to Proceed ?").upper() != "Y"):
    exit(1)

print("===== Pushing Local git to master =======")
subprocess.run(["git", "push", "origin", "master"], check=True)

try:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect("login.discovery.neu.edu", username="jain.sar")

    print("======== Running on Cluster =========")
    stdin, stdout, stderr = client.exec_command(command)
    for line in stdout:
        print("... " + line.strip("\n"))

    import time

    time.sleep(5)

    client.close()
except:
    print("Error")
    client.close()

