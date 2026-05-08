from nexus.api.main import app
for route in app.routes:
    print(f"{route.methods} {route.path}")
