#include <cstring>
#include <ctime> 
#include <deque>
#include <iostream>
#include <cstdlib>
#include <windows.h>
#include <conio.h> 
#include <stdio.h>
#include <cmath>
using namespace std;

double random(double start, double end)
{
    return start + (end - start) * rand() / (RAND_MAX + 1.0);
} //�����������[min,max)֮����� 

class Station
{

private:
    const static int max_size = 20;
    const static int max_char = 10; 
    char character[max_char] = {' ', '_', '@', '*', 'S', 'A'};
    //match none, player, coin, bomb and so on
    int num_character = 4; 
    int height;
    int width;
    //the height and width are both 10 in default.
public:
    class object
    {
    private:
        int Char;
        int x, y;
        //x match width
        //y match height
    public:
        bool legal(Station &s)
        {
            if (Char < 1 || Char >= s.num_character)
            {
                return false;
            }
            if (x < 0 || x >= s.width)
                return false;
            if (y < 0 || y >= s.height)
                return false;
            return true;
        }
    };
    HANDLE hOut;
    int caught[max_char];
    int not_caught[max_char];
    int player_x;
    int frame;
    int table[max_size][max_size];
    //for the futher design, we use the table to restore the information
    Station(int h = 10, int w = 10)
    {
        height = h;
        width = w;
        player_x = width / 2;
        frame = 0;
        memset(table, 0, sizeof(table));
        memset(caught, 0, sizeof(caught));
        memset(not_caught, 0, sizeof(not_caught));
        table[player_x][0] = 1; 
        srand(unsigned(time(0)));
    }
    bool create()
    {
        int C = int(random(2, 4));
        //C may be 2 or 3
        int x;
        int i = 0;
        do
        {
            i++;
            x = int(random(0, width));
        } while (table[x][height - 1] != 0 && i < 100);
        if (i > 100)
            return false;
        //seldom no place to display the prop
        table[x][height - 1] = C;
        return true;
    }
    void Fall()
    {
        //reflash by y increasing 
        //here is the simplest way that command the props fall directly.
        //second deal with y=1,to count the socore
        for (int i = 0; i < width; i++)
        {
            if (table[i][1] == 0) 
                continue;
            if (i == player_x)
            {
                caught[table[i][1]]++;
            }
            not_caught[table[i][1]]++;
            table[i][1] = 0; 
        }
        //other y except y=height-1
        for (int j = 2; j < height; j++)
        {
            for (int i = 0; i < width; i++)
            {
                table[i][j - 1] = table[i][j]; 
            }
        }
        for (int i = 0; i < width; i++)
        {
            table[i][height - 1] = 0;
        }
        //make new props. 
        //2/5:2,2/5:1,1/5:0�����д����µ��ߵĸ���Ϊ2�ĸ���Ϊ2/5��1�ĸ���Ϊ2/5��0�ĸ���Ϊ1/5 
        int new_prop = int(random(0, 5));
        if (new_prop < 1)
        {
            new_prop = 0;
        }
        else if (new_prop < 3)
        {
            new_prop = 1;
        }
        else
        {
            new_prop = 2;
        }
        for (int i = 0; i < new_prop; ++i)
        {
            if (!create())
            {
                cout << "new prop creation error\n";
            }
        }
    }
    int score()
    {
        //�Ʒ��� 
        return 2 * caught[2] - 1 * caught[3];
    }
    void cle(COORD ClPos)
    {
        SetConsoleCursorPosition(hOut, ClPos);
        for (int j = 0; j < height ; j++)
        {
            cout << "                                 \n";
        }
        return;
    }
    void prin(COORD ClPos)
    {
        SetConsoleCursorPosition(hOut, ClPos);
        for (int j = height - 1; j >= 0; j--)
        {
            for (int i = 0; i < width; i++)
            {
                cout << character[table[i][j]];
            }
            cout << endl;
        }
        //todo.//TODO
        cout << "coins caught:" << caught[2] << endl;
        cout << "bombs caught:" << caught[3] << endl;
        cout << "score:" << score() << "     " << endl;
        cout << endl;
        return;
    }
    //todo.//TODO
    void Move()//����ƶ� 
    {
        table[player_x][0] = 0;
        int flag = 0;//flag == 1 ����������֮�ڿ��Խӵ���� 
        for(int i = 0;i < width ; i++) 
        {
        	if(table[i][1] == 2 && abs(player_x - i ) <= 3)	
			{
				flag = 1;
				player_x = i;
				break;
			}
		}
        if(flag == 1)
	    {
	   		table[player_x][0] = 1;//����ȷ�����λ�� 
   	    }
   	    else
   	    {
   	    	do
			{
   	    		player_x = player_x + random(-3,3);
   	    		if(player_x <0)
   	    		{
   	    			player_x = 0;
				}
				else if(player_x >= width)	player_x = width-1;
			}while(table[player_x][1] == 3); 
			table[player_x][0] = 1;
   	    }
   }
    void Start()
    {
    	cout<<"this is a bot that only catches coins" << endl;
        hOut = GetStdHandle(STD_OUTPUT_HANDLE); //ȡ��� 
        COORD CrPos = {0, 1};                   //��������Ϣ 
        frame = 0; 
        clock_t start;
        start = clock();
        //todo.//TODO
        while (1)
        {
            if (clock() - start > CLOCKS_PER_SEC / 2)
            {
                Fall();
                frame++;
                start = clock();
                cle(CrPos); //���ԭ����� 
                prin(CrPos);
                Move();
				cle(CrPos); //���ԭ����� 
            	prin(CrPos);
            }
        }
    }
};
int main()
{
    SetConsoleTitle("catch it");
    Station new_game;
    new_game.Start();
    return 0;
}
