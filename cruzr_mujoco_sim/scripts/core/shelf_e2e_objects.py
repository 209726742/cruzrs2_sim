#!/usr/bin/env python3
"""Shared MuJoCo object topology helpers for rigid and articulated objects."""

import numpy as np


def _descendant_bodies(model, root_body):
    bodies = []
    for body in range(1, model.nbody):
        current = body
        while current:
            if current == root_body:
                bodies.append(body)
                break
            current = int(model.body_parentid[current])
    return tuple(bodies)


def object_info(model, name):
    """Describe one free object, including all articulated descendants."""
    import mujoco

    root_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if root_body <= 0:
        raise RuntimeError(f"scene is missing object body {name}")

    bodies = _descendant_bodies(model, root_body)
    body_set = set(bodies)
    geoms = {geom for geom in range(model.ngeom) if int(model.geom_bodyid[geom]) in body_set}

    joints = []
    for body in bodies:
        first = int(model.body_jntadr[body])
        count = int(model.body_jntnum[body])
        joints.extend(range(first, first + count))

    free_joints = [
        joint for joint in joints
        if int(model.jnt_bodyid[joint]) == root_body
        and int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if len(free_joints) != 1:
        raise RuntimeError(f"object body {name} must have exactly one root freejoint")
    ball_joints = [
        joint for joint in joints
        if int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_BALL)
    ]

    mass_kg = float(np.sum(model.body_mass[list(bodies)]))
    if mass_kg <= 0.0:
        raise RuntimeError(f"object body {name} has no positive subtree mass")
    return {
        "body": root_body,
        "bodies": bodies,
        "geoms": geoms,
        "mass_kg": mass_kg,
        "weight_n": mass_kg * abs(float(model.opt.gravity[2])),
        "free_qpos_adr": int(model.jnt_qposadr[free_joints[0]]),
        "ball_joints": tuple(ball_joints),
        "ball_joint_names": tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            for joint in ball_joints
        ),
        "ball_qpos_adrs": tuple(int(model.jnt_qposadr[joint]) for joint in ball_joints),
    }


def root_pose(data, info):
    """Return the compatibility 7D root position/quaternion pose."""
    body = info["body"]
    return np.concatenate([data.xpos[body], data.xquat[body]]).copy()


def subtree_com(model, data, info):
    """Return the world-space centre of mass of the full object subtree."""
    bodies = list(info["bodies"])
    masses = np.asarray(model.body_mass[bodies], dtype=np.float64)
    return np.sum(data.xipos[bodies] * masses[:, None], axis=0) / float(np.sum(masses))


def internal_ball_quaternions(data, info):
    """Flatten all internal ball-joint quaternions in model order."""
    addresses = info["ball_qpos_adrs"]
    if not addresses:
        return np.empty(0, dtype=np.float64)
    return np.concatenate([data.qpos[address:address + 4] for address in addresses]).copy()
