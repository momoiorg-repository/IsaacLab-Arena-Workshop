#!/bin/bash
# set -x
set -e

# Make sure that all shared libs are found.
ldconfig

# Fetch and export the public IP (used for WebRTC/LIVESTREAM access from Brev)
export PUBLIC_IP=$(curl -s ifconfig.me)
echo "Public IP: ${PUBLIC_IP}"

# --- REMOVE CONFLICTING USERS FIRST ---
# We do this first because userdel might delete the group (if names match),
# so we want to clear the slate before setting up groups.

echo "Checking for conflicting users..."

if id -u "$DOCKER_RUN_USER_ID" >/dev/null 2>&1; then
    CONFLICT_USER=$(getent passwd "$DOCKER_RUN_USER_ID" | cut -d: -f1)
    if [ "$CONFLICT_USER" != "$DOCKER_RUN_USER_NAME" ]; then
        echo "Removing user '$CONFLICT_USER' (conflicts with Target UID $DOCKER_RUN_USER_ID)..."
        userdel -f "$CONFLICT_USER" || true
    fi
fi

# Note: userdel often deletes the same-named group too, which is why we re-create below.
if id -u "$DOCKER_RUN_USER_NAME" >/dev/null 2>&1; then
    echo "Removing pre-existing user '$DOCKER_RUN_USER_NAME' to re-create with correct IDs..."
    userdel -f "$DOCKER_RUN_USER_NAME" || true
fi

# ---  SETUP THE GROUP ---
echo "Setting up group: $DOCKER_RUN_GROUP_NAME (GID: $DOCKER_RUN_GROUP_ID)"

if getent group "$DOCKER_RUN_GROUP_ID" >/dev/null; then
    EXISTING_GROUP_NAME=$(getent group "$DOCKER_RUN_GROUP_ID" | cut -d: -f1)
    if [ "$EXISTING_GROUP_NAME" != "$DOCKER_RUN_GROUP_NAME" ]; then
        echo "Conflict: GID $DOCKER_RUN_GROUP_ID is owned by '$EXISTING_GROUP_NAME'. Renaming to '$DOCKER_RUN_GROUP_NAME'..."
        groupmod --new-name "$DOCKER_RUN_GROUP_NAME" "$EXISTING_GROUP_NAME"
    else
        echo "Group '$DOCKER_RUN_GROUP_NAME' with GID $DOCKER_RUN_GROUP_ID already exists and is correct."
    fi
else
    echo "Creating group '$DOCKER_RUN_GROUP_NAME' with GID $DOCKER_RUN_GROUP_ID..."
    groupadd --gid "$DOCKER_RUN_GROUP_ID" "$DOCKER_RUN_GROUP_NAME"
fi

# --- CREATE THE USER ---
echo "Creating user $DOCKER_RUN_USER_NAME (UID: $DOCKER_RUN_USER_ID)..."
useradd --no-log-init \
        --uid "$DOCKER_RUN_USER_ID" \
        --gid "$DOCKER_RUN_GROUP_NAME" \
        --groups sudo \
        --shell /bin/bash \
        "$DOCKER_RUN_USER_NAME"

# --- PERMISSIONS & CONFIG ---
chown "$DOCKER_RUN_USER_NAME:$DOCKER_RUN_GROUP_NAME" "/home/$DOCKER_RUN_USER_NAME"
chown "$DOCKER_RUN_USER_NAME:$DOCKER_RUN_GROUP_NAME" "$WORKDIR"

# Change the root user password (so we can su root)
echo 'root:root' | chpasswd
echo "$DOCKER_RUN_USER_NAME:root" | chpasswd

# Allow sudo without password
echo "$DOCKER_RUN_USER_NAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
touch "/home/$DOCKER_RUN_USER_NAME/.sudo_as_admin_successful"

# Copy bashrc if it exists
if [ -f /etc/bash.bashrc ]; then
    cp /etc/bash.bashrc "/home/$DOCKER_RUN_USER_NAME/.bashrc"
    chown "$DOCKER_RUN_USER_NAME:$DOCKER_RUN_GROUP_NAME" "/home/$DOCKER_RUN_USER_NAME/.bashrc"
fi

# Add the models, datasets, and eval folders if they don't exist
mkdir -p /datasets /models /eval
chown "$DOCKER_RUN_USER_NAME:$DOCKER_RUN_GROUP_NAME" /datasets /models /eval

# Create the _isaac_sim symlink if it doesn't exist
if [ ! -e "$WORKDIR/submodules/IsaacLab/_isaac_sim" ]; then
    ln -s /isaac-sim/ "$WORKDIR/submodules/IsaacLab/_isaac_sim"
fi

# Run the passed command or just start the shell as the created user
if [ $# -ge 1 ]; then
    echo "alias pytest='/isaac-sim/python.sh -m pytest'" >> /etc/bash.bashrc
    exec sudo --preserve-env -u "$DOCKER_RUN_USER_NAME" \
        -- env "HOME=/home/$DOCKER_RUN_USER_NAME" bash -ic "$@"
else
    su "$DOCKER_RUN_USER_NAME"
fi
