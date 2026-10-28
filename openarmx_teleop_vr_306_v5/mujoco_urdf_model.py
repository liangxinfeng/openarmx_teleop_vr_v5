"""Build a mesh-free, fixed-base MuJoCo model for *kinematics only*.

URDF link/joint frames are retained verbatim.  A child body is placed at its
joint's zero-pose origin and a hinge/slide acts about that body's local axis.
Thus ``origin @ joint_motion`` has the same meaning in URDF and MuJoCo.

This is deliberately not a dynamics or collision-model importer: visual and
collision meshes, transmissions, actuator properties and physical inertias are
not imported.  Small synthetic inertias merely satisfy the MuJoCo compiler.
Do not use this model for physical simulation, collision checking or hardware
velocity/torque-limit validation.  The controller retains its existing URDF
collision guard and measured hardware limits for those purposes.
"""

import math
from pathlib import Path
import xml.etree.ElementTree as ET


_SUPPORTED_TYPES = {'fixed', 'revolute', 'continuous', 'prismatic'}


def _vector(text, length, default, context):
    """Read a finite URDF vector without silently accepting malformed data."""
    try:
        values = tuple(float(part) for part in (default if text is None else text).split())
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{context}: expected {length} finite numbers') from exc
    if len(values) != length or not all(math.isfinite(value) for value in values):
        raise ValueError(f'{context}: expected {length} finite numbers')
    return values


def _numbers(values):
    return ' '.join(format(value, '.17g') for value in values)


def _rpy_quaternion(rpy):
    """URDF fixed-axis XYZ RPY -> MuJoCo scalar-first quaternion (Rz Ry Rx)."""
    roll, pitch, yaw = (value / 2.0 for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _read_tree(urdf_path):
    try:
        robot = ET.parse(Path(urdf_path)).getroot()
    except ET.ParseError as exc:
        raise ValueError(f'invalid URDF XML: {exc}') from exc
    if robot.tag != 'robot':
        raise ValueError('expected a URDF <robot> root')
    links = {}
    for link in robot.findall('link'):
        name = link.get('name')
        if not name or name in links:
            raise ValueError(f'missing or duplicate URDF link name: {name!r}')
        links[name] = link
    if not links:
        raise ValueError('URDF has no links')

    children = {name: [] for name in links}
    parent_by_child = {}
    joints = {}
    for joint in robot.findall('joint'):
        name = joint.get('name')
        if not name or name in joints:
            raise ValueError(f'missing or duplicate URDF joint name: {name!r}')
        joint_type = joint.get('type')
        if joint_type not in _SUPPORTED_TYPES:
            raise ValueError(f'{name}: unsupported URDF joint type {joint_type!r}')
        if joint.find('mimic') is not None:
            raise ValueError(
                f'{name}: mimic joints require explicit coupling and are unsupported'
            )
        parent_element, child_element = joint.find('parent'), joint.find('child')
        parent = None if parent_element is None else parent_element.get('link')
        child = None if child_element is None else child_element.get('link')
        if parent not in links or child not in links:
            raise ValueError(f'{name}: parent/child references an unknown link')
        if child in parent_by_child:
            raise ValueError(f'{child}: multiple parent joints are unsupported')
        parent_by_child[child] = parent
        children[parent].append((child, joint))
        joints[name] = joint

    roots = set(links) - set(parent_by_child)
    if len(roots) != 1:
        raise ValueError('URDF must be a connected tree with exactly one fixed base')
    root_name = next(iter(roots))
    visited = set()

    def visit(link_name):
        if link_name in visited:
            raise ValueError(f'URDF has a cycle at {link_name}')
        visited.add(link_name)
        for child_name, _ in children[link_name]:
            visit(child_name)

    visit(root_name)
    if visited != set(links):
        raise ValueError('URDF contains disconnected links or a cycle')
    return robot, links, children, root_name


def urdf_to_mjcf(urdf_path, eef_frames):
    """Return deterministic MJCF text with each EEF link as a same-named site.

    ``eef_frames`` maps arm labels to URDF link names; labels do not rename the
    sites. Revolute/prismatic limits must be finite and nondegenerate.
    Continuous joints remain unbounded single-coordinate hinges. Mimic, planar
    and floating joints are rejected instead of being treated as independent
    hinges. All URDF links become same-named MuJoCo bodies, including fixed ones.
    """
    robot, links, children, root_name = _read_tree(urdf_path)
    try:
        frame_names = set(eef_frames.values())
    except (AttributeError, TypeError) as exc:
        raise ValueError('eef_frames must map arm labels to URDF link names') from exc
    if not frame_names or any(not isinstance(name, str) for name in frame_names):
        raise ValueError('eef_frames must contain at least one URDF link name')
    missing = frame_names - set(links)
    if missing:
        raise ValueError(f'URDF is missing EEF links: {sorted(missing)}')

    mjcf = ET.Element('mujoco', model=robot.get('name', 'vr_kinematics'))
    ET.SubElement(mjcf, 'compiler', angle='radian', fusestatic='false',
                  inertiafromgeom='false', autolimits='false')
    option = ET.SubElement(mjcf, 'option', gravity='0 0 0')
    ET.SubElement(option, 'flag', contact='disable')
    world = ET.SubElement(mjcf, 'worldbody')

    def append_link(parent_body, link_name, joint=None):
        body_attributes = {'name': link_name}
        if joint is not None:
            joint_name = joint.get('name')
            origin = joint.find('origin')
            xyz = _vector(None if origin is None else origin.get('xyz'),
                          3, '0 0 0', f'{joint_name} origin xyz')
            rpy = _vector(None if origin is None else origin.get('rpy'),
                          3, '0 0 0', f'{joint_name} origin rpy')
            body_attributes.update(pos=_numbers(xyz), quat=_numbers(_rpy_quaternion(rpy)))
        body = ET.SubElement(parent_body, 'body', body_attributes)
        # Dynamics are not used; never mistake URDF mesh COM for the link frame.
        ET.SubElement(body, 'inertial', pos='0 0 0', mass='0.001',
                      diaginertia='0.000001 0.000001 0.000001')
        if joint is not None and joint.get('type') != 'fixed':
            joint_type = joint.get('type')
            axis_element = joint.find('axis')
            axis = _vector(None if axis_element is None else axis_element.get('xyz'),
                           3, '1 0 0', f'{joint_name} axis')
            norm = math.sqrt(sum(value * value for value in axis))
            if not math.isfinite(norm) or norm <= 1e-12:
                raise ValueError(f'{joint_name}: joint axis must be nonzero')
            attributes = {
                'name': joint_name,
                'type': 'slide' if joint_type == 'prismatic' else 'hinge',
                'axis': _numbers(value / norm for value in axis),
                'pos': '0 0 0',
                'limited': 'false' if joint_type == 'continuous' else 'true',
            }
            if joint_type != 'continuous':
                limit = joint.find('limit')
                if limit is None or 'lower' not in limit.attrib or 'upper' not in limit.attrib:
                    raise ValueError(f'{joint_name}: missing lower/upper joint limits')
                limits = _vector(f"{limit.get('lower')} {limit.get('upper')}",
                                 2, '', f'{joint_name} limits')
                if limits[0] >= limits[1]:
                    raise ValueError(f'{joint_name}: lower limit must be below upper limit')
                attributes['range'] = _numbers(limits)
            ET.SubElement(body, 'joint', attributes)
        if link_name in frame_names:
            ET.SubElement(body, 'site', name=link_name, pos='0 0 0',
                          quat='1 0 0 0', size='0.001', rgba='0 0 0 0')
        for child_name, child_joint in children[link_name]:
            append_link(body, child_name, child_joint)

    append_link(world, root_name)
    return ET.tostring(mjcf, encoding='unicode')


def build_mujoco_model(urdf_path, eef_frames):
    """Build ``mujoco.MjModel`` without viewer, mesh loading or simulation."""
    import mujoco

    model = mujoco.MjModel.from_xml_string(urdf_to_mjcf(urdf_path, eef_frames))
    if model.nq != model.nv:
        raise ValueError('expected only fixed-base scalar joints in the MuJoCo model')
    for link_name in set(eef_frames.values()):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, link_name)
        if site_id < 0:
            raise ValueError(f'MuJoCo compiler omitted EEF site {link_name}')
    return model
