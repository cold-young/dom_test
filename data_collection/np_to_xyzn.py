import numpy as np

def extractTriangleSurfaceFromTetra(tetra_points, tetra_indices):

    # extract all triangles
    triangles = [(-1, -1, -1)] * len(tetra_indices)  # tetra has as many triangles as vertices
    for t in range(0, len(tetra_indices) // 4):
        (v0, v1, v2, v3) = (
            tetra_indices[t * 4],
            tetra_indices[t * 4 + 1],
            tetra_indices[t * 4 + 2],
            tetra_indices[t * 4 + 3],
        )
        triangles[t * 4 + 0] = (v0, v1, v2)
        triangles[t * 4 + 1] = (v1, v3, v2)
        triangles[t * 4 + 2] = (v0, v3, v1)
        triangles[t * 4 + 3] = (v0, v2, v3)

    # extract surface triangles
    surface_triangles_dict = {}
    for i, t in enumerate(triangles):
        vs = sorted([t[0], t[1], t[2]])
        key = (vs[0], vs[1], vs[2])
        if key in surface_triangles_dict:
            del surface_triangles_dict[key]
        else:
            surface_triangles_dict[key] = t

    surface_triangles = list(surface_triangles_dict.values())

    points = []
    indices = []
    tetra_points_to_points = [-1] * len(tetra_points)
    for t in surface_triangles:
        (v0, v1, v2) = t
        if tetra_points_to_points[v0] < 0:
            tetra_points_to_points[v0] = len(points)
            points.append(tetra_points[v0])
        if tetra_points_to_points[v1] < 0:
            tetra_points_to_points[v1] = len(points)
            points.append(tetra_points[v1])
        if tetra_points_to_points[v2] < 0:
            tetra_points_to_points[v2] = len(points)
            points.append(tetra_points[v2])

        indices.extend([tetra_points_to_points[v0], tetra_points_to_points[v1], tetra_points_to_points[v2]])

    return points, indices

def getNormals(points, indices):
    normals = np.zeros_like(points)
    for t in range(len(indices) // 3):
        (p0, p1, p2) = (indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2])
        v01 = points[p1]-points[p0]
        v12 = points[p2]-points[p1]
        v20 = points[p0]-points[p2]
        normals[p0] += np.cross(v01, -v20)
        normals[p1] += np.cross(v12, -v01)
        normals[p2] += np.cross(v20, -v12)
    
    for normal in normals:
        normal /= np.linalg.norm(normal)
    
    return normals

data = np.load('./Oring.npy')
index = np.load('./Oring_Indices.npy')

FOLDERNAME = "oring"
(TIMELENGTH, NUMPOINTS, _) = data.shape

for time in range(TIMELENGTH):
    p, i = extractTriangleSurfaceFromTetra(data[time,...],index)
    normal = getNormals(p,i)
    final = np.concatenate((np.array(p), normal), axis=1)
    # np.savetxt("./"+FOLDERNAME+"/"+str(time)+".xyzn", final)
    print("Processing : " + str(time) + "/" + str(TIMELENGTH))
print("Done")