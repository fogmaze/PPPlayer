#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

struct node {
    int d;
    node* next = nullptr;
    node* prev = nullptr;
    void inv() {
        node* tmp = next;
        next = prev;
        prev = tmp;
    }
};


int main() {
    vector<node*> all_trains = {};
    int N, M;
    cin >> N >> M;
    for(int i = 0; i < N; i++) {
        int k;
        cin >> k;
        if(k == 0) {
            all_trains.push_back(nullptr);
            continue;
        }
        node* nn = new node();
        cin >> nn->d;
        node* curr = nn;
        all_trains.push_back(nn);
        for(int j = 1;j < k;j++) {
            node* nn = new node();
            cin >> nn->d;
            curr->next = nn;
            nn->prev = curr;
            curr = nn;
        }
    }
    for(int i = 0;i < M; i++) {
        int o;
        cin >> o;
        if(o == 1) {
            int x;
            cin >> x;
            node* curr = all_trains[x-1];
            if(curr == nullptr) continue;
            while(curr->next != nullptr) {
                curr->inv();
                curr = curr->prev;
            }
            curr->inv();
            all_trains[x-1] = curr;
        }
        if(o == 2) {
            int a, b;
            cin >> a >> b;
            node* ah = all_trains[a-1];

            if(ah == nullptr) continue;
            if(ah->next == nullptr) continue;
            node* bl = all_trains[b-1];
            if(bl == nullptr) {
                all_trains[b-1] = ah->next;
                ah->next = nullptr;
            }
            while(bl->next != nullptr) {
                cout << 'n' << endl;
                bl = bl->next;
            }
            bl->next = ah->next;
            ah->next = nullptr;
            bl->next->prev = bl;
        }
        if(o == 3) {
            int x;
            cin >> x;
            node* curr = all_trains[x-1];
            for (; curr->next != nullptr; curr = curr->next) {
                cout << curr->d << ' ';
            }
            cout << curr->d;
            cout << '\n';
        }
    }
    return 0;
}
