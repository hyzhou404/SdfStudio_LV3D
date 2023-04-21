import numpy as np
import plotly.graph_objs as go
import torch


class Vis():
    def __init__(self, xlim=None, ylim=None, zlim=None) -> None:
        self.mesh = None
        self.ray = None
        self.fig = go.Figure()
        self.fig.update_layout(scene=dict(
            xaxis=dict(range=xlim),
            yaxis=dict(range=ylim),
            zaxis=dict(range=zlim)
        ))
        return

    def draw_bbx(self, vertices, filename=None):
        vertices = vertices.squeeze()
        num_bbx = vertices.shape[0]
        # self.max_boarder = vertices.max()
        ## i, j and k give the vertices of triangles (代表了连接顺序) 发现0123 0257 4567是一个面
        ## 每次取出 i j,k 的元素组成一个三角形，因此012 123 两个三角形组成了长方体的一个面
        i = [0, 3, 2, 5, 4, 5, 1, 1, 3, 3, 1, 1]
        j = [1, 2, 5, 0, 5, 6, 6, 6, 7, 7, 5, 5]
        k = [2, 1, 7, 2, 6, 7, 4, 3, 2, 6, 0, 4]
        for idx in range(num_bbx):
            self.fig.add_trace(
                go.Mesh3d(
                    x=vertices[idx, :, 0],
                    y=vertices[idx, :, 1],
                    z=vertices[idx, :, 2],
                    i=i,
                    j=j,
                    k=k,
                    opacity=0.1,
                    color='blue',
                    # flatshading = True,
                    name=f"bbx_{idx}"
                )
            )

        if filename is not None:
            self.vis(outfile=filename)

    def draw_ray(self, pts, output_name="vis_ray.html", selected_index=[-1]):
        random_idx = np.random.randint(low=0, high=pts.shape[0], size=50)

        if isinstance(pts, torch.Tensor):
            pts = pts.detach().cpu().numpy()
        if isinstance(selected_index,torch.Tensor):
            selected_index = selected_index.detach().cpu().numpy()

        random_idx[:10] = selected_index[0:100:10]

        for i in random_idx:
            if i in selected_index:
                color = 'red'
                print(f"Ray_{i} has intersected with aabb")
            else:
                color = "blue"

            self.fig.add_trace(
                go.Scatter3d(x=pts[i, 2:-3:2, 0],
                             y=pts[i, 2:-3:2, 1],
                             z=pts[i, 2:-3:2, 2],
                             mode='markers+lines',
                             name=f"Ray_{i}",
                             marker=dict(size=2),
                             line=dict(
                                 color=color
                             )
                             )
            )

        self.vis(outfile=output_name)

    def draw_rays_near_far(self, rays_o, nears, fars, output_name="vis_intersection.html", indices=None):
        if len(indices) < 5:
            return
        idx = indices.squeeze()[:30]
        pts_nears = nears[idx]
        pts_fars = fars[idx]
        rays_o = rays_o[idx]
        pts = torch.stack((rays_o,pts_nears,pts_fars),dim=1)
        pts = torch.stack((pts_nears, pts_fars), dim=1)
        if isinstance(pts, torch.Tensor):
            pts = pts.detach().cpu().numpy()

        for i in range(len(idx)):
            self.fig.add_trace(
                go.Scatter3d(x=pts[i, :, 0], y=pts[i, :, 1], z=pts[i, :, 2],
                             mode='markers+lines',
                             marker=dict(size=2)
                             )
            )

        # self.vis(outfile = output_name)

    def draw_random_ray(self, pts, output_name="vis__random_ray.html", selected_index=[-1],pts_id = None):
        if isinstance(pts, torch.Tensor):
            pts = pts.detach().cpu().numpy()
        if isinstance(selected_index, torch.Tensor):
            selected_index = selected_index.detach().cpu().numpy()

        for i in pts_id:
            if i in selected_index:
                color = 'red'
                print(f"Ray_{i} has intersected with aabb")
            else:
                color = "blue"

            self.fig.add_trace(
                go.Scatter3d(x=pts[i, 0:-3:2, 0],
                             y=pts[i, 0:-3:2, 1],
                             z=pts[i, 0:-3:2, 2],
                             mode='markers+lines',
                             name=f"Ray_{i}",
                             marker=dict(size=2),
                             line=dict(
                                 color=color
                             )
                             )
            )

        self.vis(outfile=output_name)

    def draw_all_ray(self, pts, output_name="vis_ray.html"):

        if isinstance(pts, torch.Tensor):
            pts = pts.detach().cpu().numpy()

        for i in range(pts.shape[0]):
            self.fig.add_trace(
                go.Scatter3d(x=pts[i, 0:-10:2, 0],
                             y=pts[i, 0:-10:2, 1],
                             z=pts[i, 0:-10:2, 2],
                             mode='markers+lines',
                             name=f"Ray_{i}",
                             marker=dict(size=2),
                             line=dict(
                                 color="green"
                             )
                             )
            )

        self.vis(outfile=output_name)

    def vis(self, outfile="out_put.html"):

        # fig = go.Figure(data=[self.mesh])
        # 显示图形

        self.fig.write_html(outfile)
        print(f"Saved to {outfile}!")

