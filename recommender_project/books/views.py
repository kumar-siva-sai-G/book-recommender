# books/views.py

from django.shortcuts import render
from .recommender import recommend_books, get_title_by_index, book_titles

def index(request):
    context = {'titles': book_titles}

    if request.method == "POST":
        action = request.POST.get('action')

        if action == 'get_title':
            try:
                index = int(request.POST.get('book_index'))
                title = get_title_by_index(index)
                context['result_title'] = title
            except (ValueError, IndexError):
                context['error'] = "Invalid index. Please try again."

        elif action == 'recommend':
            book_name = request.POST.get('book_name')
            if book_name:
                try:
                    recommendations = recommend_books(book_name)
                    context['search'] = book_name
                    context['recommendations'] = recommendations
                except ValueError:
                    context['error'] = "Sorry! Could not generate recommendations. Please check the book title."
            else:
                context['error'] = "Please enter a book name."

    return render(request, 'books/index.html', context)