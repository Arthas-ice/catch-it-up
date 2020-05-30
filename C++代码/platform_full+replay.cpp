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
    4.25����
    �Ż�ȫ�ֱ��� done
    �������� done
    �������ݸ��� �Ȳ���
    �����Ч done
*/

double random(double start, double end)
{
    return start + (end - start) * rand() / (RAND_MAX + 1.0);
}

class Station
{

private:
    const static int max_size = 20;
    const static int max_char = 10;
    char character[max_char] = { ' ', '_', '@', '*', 'S', 'A' };
    //match none, player, coin, bomb and so on
    int num_character = 4;
    int height;
    int width;
    int ID;
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
    Station(int h = 10, int w = 10)
    {
        //TODO:4.26
        //here we use a global data to initialize the platform
        ifstream input("global.dat", ios::in);
        if (!input)
        {
            cout << "default initialize\n";
            height = h;
            width = w;
            ID = 1;
        }
        else
        {
            input >> height >> width >> ID;
        }
        input.close();
        ofstream output("global.dat", ios::out);
        output << height << " " << width << " " << ID + 1;
        output.close();
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
                PlaySound(TEXT("./sounds/prop.wav"), NULL, SND_FILENAME | SND_ASYNC);
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
        //�Ʒ���
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
        hOut = GetStdHandle(STD_OUTPUT_HANDLE); //ȡ���
        COORD CrPos = { 0, 1 };                   //��������Ϣ
        frame = 0;
        clock_t start;
        start = clock();
        while (1)
        {
            if (clock() - start > CLOCKS_PER_SEC / 2)
            {
                //record();
                Fall();
                frame++;
                start = clock();
                cle(CrPos); //���ԭ�����
                prin(CrPos);
            }
            if (kbhit())
            {

                int key = getch();
                if (key == 32)
                {
                    cle(CrPos); //���ԭ�����
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
                else if (key == 82)//start replay if we press "R"
                {
                    cout<<"now start replaying.\n";
                    replay();
                }
                else
                {
                    Move(key);
                }
                cle(CrPos); //���ԭ�����
                prin(CrPos);
            }
        }
    }
   void record()
    {
        string filename;
        filename =to_string(ID)+ ".dat";
        fstream output;
        output.open(filename, ios::in);
        if (!output)
        {
            //the file is not created
            output.close();
            output.open(filename, ios::out);
            output << height << " " << width << "\n";
        }
        else
        {
            output.close();
            output.open(filename, ios::app);
        }
        output << frame << "\n";
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
    //todo:0510,����
    void replay()
    {
        string fileName = to_string(ID) + ".dat";
        ifstream in;
        in.open(fileName, ios::in);
        if(!in)
        {
            cout<<"fail to open file!\n";
            in.close();
        }
        else
        {
            int temp,h,w;
            in >> h;
            in >> w;
            hOut = GetStdHandle(STD_OUTPUT_HANDLE);
            COORD CrPos = { 0, 1 };
            clock_t  initial;
            initial = clock();
            while(in >> temp)
            {//��һ��tempΪframe���ǲ��õ����ݣ�
                for(int j = h - 1 ; j >= 0 ; j--)
                {
                    for(int i = 0 ; i < w ; i++)
                    {
                        in >> temp;
                        table[i][j] = temp;
                    }
                }
                while(1)
                {
                    if(clock() - initial > CLOCKS_PER_SEC / 2)
                    {
                        initial = clock();
                        cle(CrPos);
                        prin(CrPos);
                        break;
                    }
                }
            }
        }
        in.close();
    }
};
int main()
{
    SetConsoleTitle("catch it");
    Station new_game;
    new_game.Start();
    return 0;
}
