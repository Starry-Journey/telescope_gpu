"""Simplified 2.5 m primary-mirror FEA (Abaqus 2025, noGUI).

This is deliberately a *plate-level* model, not the full telescope:
  - annular Zerodur disk: OD 2500 mm, ID 700 mm, thickness 260 mm
  - 36 bottom support nodes (6-12-12-6)
  - 24 side support nodes (radial constraint)
  - gravity 0 deg / 90 deg
  - outputs top-surface U3 at all top nodes -> CSV

Run (Windows):
  cd C:\temp\abaqus_m1
  "D:\SIMULIA\Commands\abq2025.bat" cae noGUI=scripts/abaqus_m1_simple.py

The result is a real Abaqus solve, but the support/contact model is simplified.
Replace data/fea_cases/M1_*.csv only after checking mesh/BC convergence.
"""
from __future__ import print_function

import csv
import math
import os

from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup

executeOnCaeStartup()
Mdb()

OUT = r"C:\temp\abaqus_m1"
if not os.path.isdir(OUT):
    os.makedirs(OUT)

R_OUT = 1250.0
R_IN = 350.0
THICK = 260.0
E = 90000.0          # MPa
NU = 0.24
RHO = 2.53e-9        # t/mm^3


def nearest_node(inst, x, y, z, radius=25.0):
    nodes = inst.nodes.getByBoundingSphere(center=(x, y, z), radius=radius)
    if not nodes:
        nodes = inst.nodes.getByBoundingSphere(center=(x, y, z), radius=radius * 3)
    if not nodes:
        raise RuntimeError("no node near %s,%s,%s" % (x, y, z))
    return nodes[0]


def build_model(tag, gx, gz):
    name = "M1_" + tag
    if name in mdb.models:
        del mdb.models[name]
    model = mdb.Model(name=name)
    if "Model-1" in mdb.models and mdb.models["Model-1"].name != name:
        del mdb.models["Model-1"]

    # annular sketch -> solid
    s = model.ConstrainedSketch(name="profile", sheetSize=3000.0)
    s.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(R_OUT, 0.0))
    s.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(R_IN, 0.0))
    part = model.Part(name="M1", dimensionality=THREE_D, type=DEFORMABLE_BODY)
    part.BaseSolidExtrude(sketch=s, depth=THICK)

    # material/section
    mat = model.Material(name="Zerodur")
    mat.Elastic(table=((E, NU),))
    mat.Density(table=((RHO,),))
    model.HomogeneousSolidSection(name="M1sec", material="Zerodur", thickness=None)
    part.SectionAssignment(region=(part.cells,), sectionName="M1sec")

    # mesh
    part.seedPart(size=45.0, deviationFactor=0.2, minSizeFactor=0.2)
    elem = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD)
    part.setElementType(regions=(part.cells,), elemTypes=(elem,))
    part.generateMesh()

    asm = model.rootAssembly
    asm.DatumCsysByDefault(CARTESIAN)
    inst = asm.Instance(name="M1-1", part=part, dependent=ON)

    # bottom supports: 6-12-12-6
    bottom = []
    for radius, count in ((375.0, 6), (625.0, 12), (875.0, 12), (1125.0, 6)):
        for k in range(count):
            ang = 2.0 * math.pi * k / count
            bottom.append(nearest_node(inst, radius * math.cos(ang), radius * math.sin(ang), 0.0))
    asm.Set(name="BOTTOM_SUPPORTS", nodes=bottom)
    model.EncastreBC(name="FixBottom", createStepName="Initial",
                     region=asm.sets["BOTTOM_SUPPORTS"])

    # side supports: 24 radial constraints at mid-thickness
    side = []
    for k in range(24):
        ang = 2.0 * math.pi * k / 24.0
        side.append(nearest_node(inst, R_OUT * math.cos(ang), R_OUT * math.sin(ang), THICK / 2.0))
    asm.Set(name="SIDE_SUPPORTS", nodes=side)
    model.DisplacementBC(name="FixSide", createStepName="Initial",
                         region=asm.sets["SIDE_SUPPORTS"],
                         u1=0.0, u2=0.0, u3=UNSET)

    # gravity in mm/s^2
    model.StaticStep(name="gravity", previous="Initial", timePeriod=1.0)
    model.Gravity(name="g", createStepName="gravity", comp1=gx, comp2=0.0, comp3=gz)
    model.fieldOutputRequests["F-Output-1"].setValues(variables=("U",))

    job = mdb.Job(name=name, model=name, type=ANALYSIS, numCpus=1)
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    return name


def export_top(name):
    odb_path = os.path.join(OUT, name + ".odb")
    odb = openOdb(odb_path, readOnly=True)
    inst = odb.rootAssembly.instances["M1-1"]
    frame = odb.steps["gravity"].frames[-1]
    u = frame.fieldOutputs["U"]
    rows = []
    for v in u.values:
        n = inst.nodes[v.nodeLabel - 1]
        x, y, z = n.coordinates
        if z > THICK - 5.0:
            rows.append((x, y, z, v.data[0], v.data[1], v.data[2]))
    with open(os.path.join(OUT, name + "_top.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["x_mm", "y_mm", "z_mm", "u1_mm", "u2_mm", "dz_nm"])
        for x, y, z, u1, u2, u3 in rows:
            wr.writerow([x, y, z, u1, u2, u3 * 1.0e6])
    odb.close()


for tag, gx, gz in (("0deg", 9810.0, 0.0), ("90deg", 0.0, -9810.0)):
    print("building", tag)
    name = build_model(tag, gx, gz)
    export_top(name)
print("DONE")
