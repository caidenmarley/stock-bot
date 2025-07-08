#include "parser/parser.h"

int main() {
    CSVLoader loader("data/AAAU.csv");
    const auto& rows = loader.getData();

    for (size_t i = 0; i < 10 && i < rows.size(); ++i) {
        const auto& r = rows[i];
        std::cout << "Date: " << r.date << " Open: " << r.open << "\n";
    }
}
