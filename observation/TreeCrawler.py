import math
from fl_base.observations import Node

# Utils class to enumerate paths by tree nodes

class TreeCrawler():
    node_keys = ["L", "F", "R", "B"]

    # def crawl(tree:Node, target: Tuple[int,int]):
    def crawl(self, root_node: Node):
        # enumerates all paths to target starting at root node

        # start at root
        self.paths = []

        for k, v in root_node.childs.items():
            if v != -math.inf:
                self.eval_child_branch(k, root_node, [])
        if False:
            for p in self.paths:
                s = str(p[0])
                for i in p[1]:
                    s = s + " => " + str(i[0]) + ": " + str(i[1])
                print(s)

        return self.paths

    def eval_child_branch(self, key, root: Node, path):
        if len(root.childs) == 0:
            return

        local_path = path.copy()
        local_path.append([root.node_pos, key, root])
        cur_node = root.childs[key]

        # new -> to eval this SECTION/Branch there must be no conflict
        if 0 < cur_node.potential_conflict_dist < math.inf:
            return

        for k, v in cur_node.childs.items():
            candidate_node: Node = v
            if candidate_node != -math.inf:
                if candidate_node.dist_own_target_encountered != math.inf:
                    p = local_path.copy()
                    p.append([cur_node.node_pos, k, cur_node])
                    # candiate is target -> k belongs to cur_node-> refers to action taken at node x
                    p.append([candidate_node.node_pos, k, candidate_node])
                    self.paths.append([candidate_node.dist_own_target_encountered, p])
                    continue

                if candidate_node.potential_conflict_dist == 0 or candidate_node.potential_conflict_dist == math.inf:
                    self.eval_child_branch(k, cur_node, local_path)


    def get_node_data(self, key, node: Node):
        return {key: [node, node.dist_min_to_target]}

    # def copy(self, n: Node):
    #     clone = Node(dist_own_target_encountered=n.dist_own_target_encountered,
    #                  dist_other_target_encountered=n.dist_other_target_encountered,
    #                  dist_other_agent_encountered=n.dist_other_agent_encountered,
    #                  potential_conflict_dist=n.potential_conflict_dist,
    #                  potential_conflict_pos=n.potential_conflict_pos,
    #                  potential_conflict_time_step=n.potential_conflict_time_step,
    #                  dist_unusable_switch=n.dist_unusable_switch,
    #                  dist_to_next_branch=n.dist_to_next_branch,
    #                  dist_min_to_target=n.dist_min_to_target,
    #                  num_agents_same_direction=n.num_agents_same_direction,
    #                  num_agents_opposite_direction=n.num_agents_opposite_direction,
    #                  num_agents_malfunctioning=n.num_agents_malfunctioning,
    #                  speed_min_fractional=n.speed_min_fractional,
    #                  num_agents_ready_to_depart=n.num_agents_ready_to_depart,
    #                  node_pos=n.node_pos,
    #                  agent_direction=n.agent_direction,
    #                  childs={})
    #     for k, v in n.childs.items():
    #         clone.childs[k] = v
    #
    #     return clone
