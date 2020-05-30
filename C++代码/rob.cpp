#include <cstring>
#include <ctime>
#include <deque>
#include <iostream>
#include <cstdlib>
#include <windows.h>
#include <conio.h>
#include <stdio.h>
#include <fstream>
#include <string>
#include<windows.h>
#include<Mmsystem.h>
#pragma comment(lib, "winmm.lib")
using namespace std;
/*
    4.25更新
    优化全局变量 done
    保存数据 done
    根据数据复盘 先不做
    添加声效 done
*/

double random(double start, double end)
{
    return start + (end - start) * rand() / (RAND_MAX + 1.0);
}

class Station
{
public:
    const static int max_size = 20;
    const static int max_char = 10;
    char character[max_char] = { ' ', '_', '@', '*', 'S', 'A' };
    //match none, player, coin, bomb and so on
    int num_character = 4;
    int height;
    int width;
    int ID;
    //the height and width are both 10 in default.
    class object
    {
    private:
        int Char;
        int x, y;
        //x match width
        //y match height
    public:
        bool legal(Station& s)
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
    //the player is in place(player_x, 0)
    int table[max_size][max_size];
    //for the futher design, we use the table to restore the information
    Station(int h = 16, int w = 16)
    {
        //TODO:4.26
        //here we use a global data to initialize the platform
 
        cout << "default initialize\n";
        height = h;
        width = w;
        ID = 1;
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
                //PlaySound(TEXT("./sounds/prop.wav"), NULL, SND_FILENAME | SND_ASYNC);
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
                //perfect
            }
        }
        for (int i = 0; i < width; i++)
        {
            table[i][height - 1] = 0;
        }
        //make new props.
        //2/5:2,2/5:1,1/5:0
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
        //计分器
        return 2 * caught[2] - 1 * caught[3];
    }
    void cle(COORD ClPos)
    {
        SetConsoleCursorPosition(hOut, ClPos);
        for (int j = 0; j < height; j++)
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
        cout << "score:" << score() << "     " << endl;
        cout << endl;
        return;
    }
    void Move(int key)
    {
        table[player_x][0] = 0;
        switch (key)
        {
        case 75:
            PlaySound(TEXT("./sounds/move.wav"), NULL, SND_FILENAME | SND_ASYNC);
            player_x--;
            if (player_x < 0)
                player_x = 0;
            break;
        case 77:
            PlaySound(TEXT("./sounds/move.wav"), NULL, SND_FILENAME | SND_ASYNC);
            player_x++;
            if (player_x >= width)
                player_x = width - 1;
            break;
        default:
            break;
        }
        table[player_x][0] = 1;
    }
    void Start()
    {
        cout << "left and right arrow to control, Spacebar to pause" << endl;
        hOut = GetStdHandle(STD_OUTPUT_HANDLE); //取句柄
        COORD CrPos = { 0, 1 };                   //保存光标信息
        frame = 0;
        clock_t start;
        start = clock();
        while (1)
        {
            if (clock() - start > CLOCKS_PER_SEC / 2)
            {
                record();
                Fall();
                frame++;
                start = clock();
                cle(CrPos); //清除原有输出
                prin(CrPos);
            }
            if (kbhit())
            {

                int key = getch();
                if (key == 32)
                {
                    cle(CrPos); //清除原有输出
                    SetConsoleCursorPosition(hOut, CrPos);
                    cout << "pause,enter Spacebar to continue\n";
                    while (1)
                    {
                        if (kbhit())
                        {
                            if (getch() == 32)
                            {
                                break;
                            }
                        }
                    }
                }
                else
                {
                    Move(key);
                }
                cle(CrPos); //清除原有输出
                prin(CrPos);
            }
        }
    }
    //TODO:4.26
    //save all the data in one document
    void record()
    {
        string filename;
        filename = to_string(ID) + ".dat";
        fstream output;
        output.open(filename, ios::in);
        if (!output)
        {
            //the file is not created
            output.close();
            output.open(filename, ios::out);
            //output << height << " " << width << "\n";
        }
        else
        {
            output.close();
            output.open(filename, ios::out);
        }

        //output << frame << "\n";
        for (int j = height - 1; j >= 0; j--)
        {
            for (int i = 0; i < width; i++)
            {
                output << table[i][j] << " ";
            }
            output << "\n";
        }
        output.close();
    }
};
class Robot_Station :public Station
{
public:
    Robot_Station(int h, int w, int id)
    {
        cout << "dsadsad";
        if (id == 0)
            return;
        string filename;
        ID = id;
        filename = "./table/"+to_string(id) + ".dat";
        fstream input;
        input.open(filename, ios::in);
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
               input>> table[j][height-1-i];
            }
        }
        for (int i = 0; i < width; i++)
        {
            table[i][0] = 0;
        }
        input.close();
        cout << "read table\n";
        filename = "./action/" + to_string(ID) + ".dat";
        int action;
        input.open(filename, ios::in);
        input >> action;
        input.close();
        table[action][0] = 1;
        player_x = action;
        cout << "read action\n";
        ID++;
    }
    void Start()
    {
        Fall();
        record(ID);
        return;
    }
    void record(int id)
    {
        string filename;
        filename = "./table/" + to_string(id) + ".dat";
        fstream output;
        output.open(filename, ios::in);
        if (!output)
        {
            //the file is not created
            output.close();
            output.open(filename, ios::out);
            //output << height << " " << width << "\n";
        }
        else
        {
            output.close();
            output.open(filename, ios::app);
        }
        //output << frame << "\n";
        for (int j = height - 1; j >= 0; j--)
        {
            for (int i = 0; i < width; i++)
            {
                output << table[i][j] << " ";
            }
            output << "\n";
        }
        output.close();
    }
};
int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        cout << "command error!\n";
        return 0;
    }
    int Id = atoi(argv[1]);
    Robot_Station stn(16, 16, Id);
    stn.Start();
    return 0;
}