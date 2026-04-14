c = get_config()

# Alpha Factory JupyterHub Configuration
c.JupyterHub.authenticator_class = "nativeauthenticator.NativeAuthenticator"
c.JupyterHub.spawner_class = "dockerspawner.DockerSpawner"
c.DockerSpawner.image = "jupyter/scipy-notebook:latest"
c.DockerSpawner.network_name = "jupyterhub_network"

# Resource limits for researchers
c.DockerSpawner.cpu_limit = 2
c.DockerSpawner.mem_limit = "4G"

# Persistence
c.DockerSpawner.remove = True
c.DockerSpawner.notebook_dir = "/home/jovyan/work"
