import math
import Agent
import Obstacle
import Cone

e = 0.01

def dist(pos0, pos1):
    return math.sqrt((pos0[0]-pos1[0])**2 + (pos0[1]-pos1[1])**2)

def angle(u, v):
    try:
        #return math.atan2(((-u[1]*v[0])+(u[0]*v[1])), (u[0]*v[0]) + (u[1]*v[1]))
        theta = math.atan2(v[1], v[0]) - math.atan2(u[1], u[0])
        if theta > math.pi:
            theta -= 2 * math.pi
        elif theta <= -math.pi:
            theta += 2 * math.pi
        return theta
    except Exception as e:
        print(e)
        return 0

def rotate(v, theta):
    length = dist([0,0], v)
    x = v[0]*math.cos(theta) - v[1]*math.sin(theta)
    y = v[0]*math.sin(theta) + v[1]*math.cos(theta)
    return [x,y]
    

def minkowski_sum(A, B):
    m_sum = set()
    for a in A:
        for b in B:
            m_sum.add((a[0] + b[0], a[1] + b[1]))
    return m_sum

def VO(A, B, pA, pB, vA, vB, gA):
    VO_AB = []
    minus_A = [(-a1, -a0) for (a0, a1) in A]
    m_sum = minkowski_sum(B, A)
    for u in m_sum:
        if dist(pB, pA) < dist(pA, gA):
            v = [u[0] + vB[0] + pB[0] - pA[0], u[1] + vB[1] + pB[1] - pA[1]]
        
            VO_AB.append(v)
    return VO_AB

def RVO(A, B, pA, pB, vA, vB, gA):
    RVO_AB = []
    VO_AB = VO(A, B, pA, pB, vA, vB, gA)
    for v in VO_AB:
        RVO_AB.append((v[0] + 0.5*(-vB[0] + vA[0]), v[1] + 0.5*(-vB[1] + vA[1])))
        #RVO_AB.append(v)
    return RVO_AB

def CRVO(agents, obstacles):
    for i in agents:
        RVOi = []
        apexes = []
        for j in agents:
            if i.agent_id != j.agent_id:
                points = list(RVO(i.poly, j.poly, i.p, j.p, i.v, j.v, i.g))
                apexes.append([(i.v[0] + j.v[0])/2,(i.v[1] + j.v[1])/2])
                RVOi.append(points)
        for o in obstacles:
            points = list(VO(i.poly, o.poly, i.p, o.p, i.v, [0.,0.], i.g))
            apexes.append([0.,0.])
            RVOi.append(points)
        i.set_CRVO(RVOi)
        i.set_apexes(apexes)

def find_velocity(agent, RVOi, apexes, v_pref, max_vel):
    #Assumes all agents and obstacles are convex
    cones = []
    for apex, points in zip(apexes, RVOi):
        if len(points) > 0:
            min_angle = math.pi
            max_angle = -math.pi
            min_point = points[0]
            max_point = points[0]
            angles = []
            u = [(points[0][0] - apex[0]), (points[0][1] - apex[1])]
            for point in points:
                moved_point = [(point[0] - apex[0]), (point[1] - apex[1])]
                angle_dif = angle(u, moved_point)
                angles.append(angle_dif)
                if angle_dif < min_angle:
                    min_angle = angle_dif
                    min_point = point
                if angle_dif > max_angle:
                    max_angle = angle_dif
                    max_point = point
            
            cones.append(Cone.Cone(apex,min_point, max_point))            
        

    within_VO = False
    for cone in cones:
        if cone.contains(v_pref):
            within_VO = True
    if not within_VO:
        return v_pref, cones, []
    
    else:  
        potential_v_left = []
        potential_v_right = []
        all_v = []
        for i in range(10,1,-1):
            for j in range(0,61):
                new_v = rotate(v_pref, j/60 * math.pi)
                magnitude = math.sqrt(new_v[0]**2 + new_v[1]**2)
                new_v_normalised = (new_v[0]/magnitude, new_v[1]/magnitude)
                new_v = [new_v_normalised[0]*i*max_vel/10,new_v_normalised[1]*i*max_vel/10]
                new_v_loc = [agent.p[0] + new_v[0],agent.p[1] + new_v[1]]
                within_VO = False
                for cone in cones:
                    if cone.contains(new_v):
                        within_VO = True
                if not within_VO:
                    potential_v_left.append(new_v)
                all_v.append(new_v)
                
                new_v = rotate(v_pref, -j/60 * math.pi)
                magnitude = math.sqrt(new_v[0]**2 + new_v[1]**2)
                new_v_normalised = (new_v[0]/magnitude, new_v[1]/magnitude)
                new_v = [new_v_normalised[0]*i*max_vel/10,new_v_normalised[1]*i*max_vel/10]
                new_v_loc = [agent.p[0] + new_v[0],agent.p[1] + new_v[1]]
                within_VO = False
                for cone in cones:
                    if cone.contains(new_v):
                        within_VO = True
                if not within_VO:
                    potential_v_right.append(new_v)    
                all_v.append(new_v)
                
                potential_v = potential_v_left + potential_v_right
    
    if len(potential_v) > 0:
        new_v = potential_v[0]
        new_v_dist = dist(new_v, v_pref)
        for v in potential_v:
            v_dist = dist(v, v_pref)
            v_angle = angle(agent.v, v)
            if new_v_dist > v_dist and abs(v_angle) < 0.5*math.pi:
                new_v_dist = v_dist
                new_v = v
        
        #new_v = potential_v[0]        

        return new_v, cones, all_v
    else:
        return (0,0), cones, all_v
