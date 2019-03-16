# utility functions for priority sweeping

import heapq

# used to maintain sorted pairs of (priority, (state,action))
class qItem():
    def __init__(self, P,s,a):
        self.item = (P,(s,a))
    def __lt__(self, other):
        return self.item[0] < other.item[0]

class pQueue():
    def __init__(self):
        self.queue = []

    # flips P back after being flipped in enqueue
    def dequeue(self):
        qitem = heapq.heappop(self.queue)
        return -qitem.item[0], qitem.item[1][0], qitem.item[1][1]
    
    # keeps at most one item per (s,a) pair
        # the one with highest priority
    # flips P to sort in decreasing order
    def enqueue(self, P,s,a):
        qitem = qItem(-P,s,a)
        index = self.find(qitem)
        if index == -1:
            heapq.heappush(self.queue, qitem)
        else:
            oldP = self.queue[index].item[0]
            if -P < oldP:
                self.queue[index] = qitem
                heapq.heapify(self.queue)

    # looks for another instance in queue with same (s,a)
    def find(self, new_q_item):
        search_list = [x.item[1] for x in self.queue]
        index = -1
        try:
            index = search_list.index(new_q_item.item[1])
        except:
            pass
        return index

