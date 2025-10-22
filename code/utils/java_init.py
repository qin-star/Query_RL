import os
import sys

def get_jdk_path():
    # Try to get home directory from environment or use a fallback
    home = os.environ.get('HOME')
    if home is None:
        # Fallback to user's home directory using os.path.expanduser
        home = os.path.expanduser('~')
        if home == '~':
            # If still no home directory, try to get it from the current user
            import pwd
            home = pwd.getpwuid(os.getuid()).pw_dir
    
    jdk_dir = os.path.join(home, ".jdk")
    
    if not os.path.exists(jdk_dir):
        raise RuntimeError(f"JDK directory not found at {jdk_dir}")
    
    # List all directories in .jdk that start with 'jdk-'
    jdk_versions = [d for d in os.listdir(jdk_dir) if d.startswith('jdk-')]
    
    if not jdk_versions:
        raise RuntimeError(f"No JDK installations found in {jdk_dir}")
    
    # Get the latest version
    latest_version = sorted(jdk_versions)[-1]
    return os.path.join(home, ".jdk", latest_version)

try:
    os.environ['JAVA_HOME'] = get_jdk_path()
    os.environ['PATH'] = os.environ['PATH'] + ':' + os.path.join(os.environ['JAVA_HOME'], 'bin')
except Exception as e:
    print(f"[Warning] Fail to set up Java environment via our provided code: {str(e)}. You could use the java env in your own machine.", file=sys.stderr)
    
    
if __name__ == "__main__":
    print(get_jdk_path())
